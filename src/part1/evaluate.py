import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from src.utils import set_seed, load_config, init_wandb, load_checkpoint, load_multilingual_tokenizer, get_device
from src.part1.model import TransformerConfig, Transformer
from src.common.decoding import greedy_decode
from src.part1.data import (
    get_translation_dataloaders,
)


def compute_test_perplexity(
    model: Transformer,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_test_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute cross-entropy loss and perplexity on test set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if max_test_batches is not None and i >= max_test_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids,
                labels=labels,
                attention_mask=batch.get("attention_mask", None).to(device)
                if batch.get("attention_mask") is not None else None,
            )
            lm_loss = out["lm_loss"]

            n_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += lm_loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 50.0))
    return avg_loss, perplexity


def generate_translation(
    model: Transformer,
    tokenizer: Any,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 64,
) -> str:
    """
    Autoregressive greedy generation from manual model forward passes
    without using HF .generate().
    """
    eos_id = tokenizer.eos_token_id or 3
    prompt_ids = tokenizer.encode(f"<bos> {prompt_text}", add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = greedy_decode(
        model,
        input_ids,
        max_new_tokens=min(max_new_tokens, model.config.max_seq_len - input_ids.size(1)),
        eos_token_id=eos_id,
        use_cache=False,
    )
    generated_ids = generated[0, len(prompt_ids):].tolist()
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return decoded


def compute_bleu_scores(
    model: Transformer,
    tokenizer: Any,
    test_samples: List[Dict[str, str]],
    device: torch.device,
    max_eval_samples: int = 200,
) -> Dict[str, float]:
    """Compute corpus BLEU scores overall and broken down by source language."""
    import sacrebleu

    eval_subset = test_samples[:max_eval_samples]
    all_refs = []
    all_hyps = []

    vi_refs = []
    vi_hyps = []

    ja_refs = []
    ja_hyps = []

    print(f"[BLEU Evaluation] Generating translations for {len(eval_subset)} test samples...")
    for item in tqdm(eval_subset, desc="Evaluating BLEU"):
        src_lang = item["src_lang"]
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]

        prompt = f"<{src_lang}> {src_text} <en>"
        hyp = generate_translation(model, tokenizer, prompt, device, max_new_tokens=64)

        all_refs.append(tgt_text)
        all_hyps.append(hyp)

        if src_lang == "vi":
            vi_refs.append(tgt_text)
            vi_hyps.append(hyp)
        elif src_lang == "ja":
            ja_refs.append(tgt_text)
            ja_hyps.append(hyp)

    overall_bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score if all_hyps else 0.0
    vi_bleu = sacrebleu.corpus_bleu(vi_hyps, [vi_refs]).score if vi_hyps else 0.0
    ja_bleu = sacrebleu.corpus_bleu(ja_hyps, [ja_refs]).score if ja_hyps else 0.0

    return {
        "bleu_overall": overall_bleu,
        "bleu_vi_en": vi_bleu,
        "bleu_ja_en": ja_bleu,
    }


def compute_and_plot_expert_heatmaps(
    model: Transformer,
    tokenizer: Any,
    test_samples: List[Dict[str, str]],
    device: torch.device,
    output_path: Path,
    max_samples: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    Compute and visualize expert routing frequencies separated by source language (vi vs ja).
    Generates layer x expert heatmaps showing language specialization across MoE layers.
    """
    if not model.config.use_moe:
        print("[Heatmap] Skipping expert heatmap for dense baseline (no MoE layers).")
        return None

    model.eval()
    n_layers = model.config.n_layers
    n_experts = model.config.n_experts

    counts_vi = np.zeros((n_layers, n_experts), dtype=np.float64)
    counts_ja = np.zeros((n_layers, n_experts), dtype=np.float64)
    total_tokens_vi = np.zeros(n_layers, dtype=np.float64)
    total_tokens_ja = np.zeros(n_layers, dtype=np.float64)

    eval_subset = test_samples[:max_samples]
    print(f"[Heatmap] Collecting routing activations on {len(eval_subset)} samples...")

    with torch.no_grad():
        for item in eval_subset:
            src_lang = item["src_lang"]
            src_text = item["src_text"]
            prompt = f"<{src_lang}> {src_text} <en>"
            input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

            _ = model(input_ids, attention_mask=torch.ones_like(input_ids))

            for l_idx, layer in enumerate(model.layers):
                moe = layer.ffn
                if hasattr(moe, "last_routing_indices") and moe.last_routing_indices is not None:
                    chosen_experts = moe.last_routing_indices.view(-1).cpu().numpy()
                    if model.config.moe_variant == "shared_plus_routed":
                        counts = counts_vi if src_lang == "vi" else counts_ja
                        counts[l_idx, 0] += len(chosen_experts)
                        chosen_experts = chosen_experts + 1
                    for exp_id in chosen_experts:
                        if src_lang == "vi":
                            counts_vi[l_idx, exp_id] += 1
                        else:
                            counts_ja[l_idx, exp_id] += 1

                    active_per_token = 2 if model.config.moe_variant == "shared_plus_routed" else model.config.top_k
                    routed_token_count = moe.last_routing_indices.size(0)
                    if src_lang == "vi":
                        total_tokens_vi[l_idx] += routed_token_count * active_per_token
                    else:
                        total_tokens_ja[l_idx] += routed_token_count * active_per_token

    freq_vi = np.divide(counts_vi, total_tokens_vi[:, None], out=np.zeros_like(counts_vi), where=total_tokens_vi[:, None] > 0)
    freq_ja = np.divide(counts_ja, total_tokens_ja[:, None], out=np.zeros_like(counts_ja), where=total_tokens_ja[:, None] > 0)
    diff = freq_vi - freq_ja

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    expert_labels = [f"Exp {i}" for i in range(n_experts)]
    layer_labels = [f"L{i+1}" for i in range(n_layers)]

    sns.heatmap(freq_vi, ax=axes[0], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=expert_labels, yticklabels=layer_labels, cbar_kws={'label': 'Activation Frequency'})
    axes[0].set_title(f"Vietnamese (vi) Expert Activation\n({model.config.moe_variant})")
    axes[0].set_xlabel("Experts")
    axes[0].set_ylabel("Layers")

    sns.heatmap(freq_ja, ax=axes[1], annot=True, fmt=".2f", cmap="Greens",
                xticklabels=expert_labels, yticklabels=layer_labels, cbar_kws={'label': 'Activation Frequency'})
    axes[1].set_title(f"Japanese (ja) Expert Activation\n({model.config.moe_variant})")
    axes[1].set_xlabel("Experts")
    axes[1].set_ylabel("Layers")

    sns.heatmap(diff, ax=axes[2], annot=True, fmt="+.2f", cmap="vlag", center=0.0,
                xticklabels=expert_labels, yticklabels=layer_labels, cbar_kws={'label': 'Frequency Diff (Vi - Ja)'})
    axes[2].set_title(f"Specialization Divergence (Vi - Ja)\n({model.config.moe_variant})")
    axes[2].set_xlabel("Experts")
    axes[2].set_ylabel("Layers")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Heatmap Saved] Expert activation heatmap saved to {output_path}")

    fraction_vi = freq_vi
    fraction_ja = freq_ja
    entropy_vi = -np.sum(np.where(fraction_vi > 0, fraction_vi * np.log(np.maximum(fraction_vi, 1e-12)), 0.0), axis=1)
    entropy_ja = -np.sum(np.where(fraction_ja > 0, fraction_ja * np.log(np.maximum(fraction_ja, 1e-12)), 0.0), axis=1)
    raw_data = {
        "variant": model.config.moe_variant,
        "expert_labels": ["shared" if model.config.moe_variant == "shared_plus_routed" and i == 0 else f"expert_{i}" for i in range(n_experts)],
        "counts_vi": counts_vi.astype(int).tolist(),
        "counts_ja": counts_ja.astype(int).tolist(),
        "token_fractions_vi": fraction_vi.tolist(),
        "token_fractions_ja": fraction_ja.tolist(),
        "load_balancing_entropy_vi": entropy_vi.tolist(),
        "load_balancing_entropy_ja": entropy_ja.tolist(),
        "per_layer": [
            {
                "layer": layer_index + 1,
                "vi_counts": counts_vi[layer_index].astype(int).tolist(),
                "ja_counts": counts_ja[layer_index].astype(int).tolist(),
                "vi_fractions": fraction_vi[layer_index].tolist(),
                "ja_fractions": fraction_ja[layer_index].tolist(),
            }
            for layer_index in range(n_layers)
        ],
    }
    raw_path = output_path.with_suffix(".json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    print(f"[Routing Counts Saved] Raw expert usage written to {raw_path}")
    table_path = output_path.with_suffix(".csv")
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "language", "expert", "selected_tokens", "token_fraction", "load_balancing_entropy"],
        )
        writer.writeheader()
        for layer_index in range(n_layers):
            for language, counts, fractions, entropies in (
                ("vi", counts_vi, fraction_vi, entropy_vi),
                ("ja", counts_ja, fraction_ja, entropy_ja),
            ):
                for expert_index, label in enumerate(raw_data["expert_labels"]):
                    writer.writerow({
                        "layer": layer_index + 1,
                        "language": language,
                        "expert": label,
                        "selected_tokens": int(counts[layer_index, expert_index]),
                        "token_fraction": float(fractions[layer_index, expert_index]),
                        "load_balancing_entropy": float(entropies[layer_index]),
                    })
    print(f"[Routing Table Saved] Expert usage table written to {table_path}")

    return {
        "freq_vi": freq_vi.tolist(),
        "freq_ja": freq_ja.tolist(),
        "diff": diff.tolist(),
        "load_balancing_entropy_vi": entropy_vi.tolist(),
        "load_balancing_entropy_ja": entropy_ja.tolist(),
        "raw_counts_path": str(raw_path),
        "routing_table_path": str(table_path),
    }


def evaluate_part1(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    profile: Optional[str] = None,
):
    config = load_config(config_path)
    if profile is not None:
        config["profile"] = profile
    if str(config.get("profile", "full")).lower() == "smoke":
        config["d_model"] = int(config.get("smoke_d_model", 64))
        config["n_layers"] = int(config.get("smoke_n_layers", 2))
        config["n_heads"] = int(config.get("smoke_n_heads", 4))
        config["d_ff"] = int(config.get("smoke_d_ff", 256))
        config["max_seq_len"] = int(config.get("smoke_max_seq_len", 64))
        config["max_samples"] = int(config.get("smoke_max_samples", 64))
        config["allow_synthetic_data"] = bool(config.get("smoke_allow_synthetic_data", True))
    set_seed(config.get("seed", 42))

    device = get_device()
    print(f"[Part 1 Eval] Using device: {device}")

    # 1. Tokenizer & Test Data
    tokenizer_dir = config.get("tokenizer_dir", "./data/tokenizer_part1")
    tokenizer = load_multilingual_tokenizer(tokenizer_dir)

    _, _, test_loader = get_translation_dataloaders(config, tokenizer)
    test_pairs = test_loader.dataset.samples

    # 2. Build Model
    model_cfg = TransformerConfig(
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        vocab_size=len(tokenizer),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,
        use_moe=config.get("use_moe", False),
        moe_variant=config.get("moe_variant", "dense"),
        n_experts=config.get("n_experts", 4),
        top_k=config.get("top_k", 1),
    )
    model = Transformer(model_cfg).to(device)

    # 3. Load Checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(config.get("checkpoint_dir", f"./runs/checkpoints/part1_{model_cfg.moe_variant}"))
        candidates = [
            checkpoint_dir / "best_model.safetensors",
            checkpoint_dir / "best_model.pt",
            checkpoint_dir / "final_model.safetensors",
            checkpoint_dir / "final_model.pt",
        ]
        checkpoint_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model=model, device=str(device))
    else:
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}; refusing to report random-model metrics."
        )

    # 4. Evaluate Perplexity
    test_loss, test_ppl = compute_test_perplexity(
        model,
        test_loader,
        device,
        max_test_batches=config.get("max_test_batches"),
    )
    print(f"\n[Part 1 Test Results] Perplexity: {test_ppl:.2f} (Loss: {test_loss:.4f})")

    # 5. Evaluate BLEU
    max_bleu_samples = config.get("max_bleu_samples", 200)
    bleu_results = compute_bleu_scores(model, tokenizer, test_pairs, device, max_eval_samples=max_bleu_samples)
    print(f"[BLEU Results] Overall: {bleu_results['bleu_overall']:.2f} | "
          f"Vi->En: {bleu_results['bleu_vi_en']:.2f} | Ja->En: {bleu_results['bleu_ja_en']:.2f}")

    # 6. Generate Expert Heatmaps
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    heatmap_path = assets_dir / f"part1_heatmap_{model_cfg.moe_variant}.png"
    compute_and_plot_expert_heatmaps(
        model=model,
        tokenizer=tokenizer,
        test_samples=test_pairs,
        device=device,
        output_path=heatmap_path,
        max_samples=config.get("max_heatmap_samples", 200),
    )

    # 7. Save Metrics Summary
    results = {
        "variant": model_cfg.moe_variant,
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "bleu": bleu_results,
        "parameter_stats": model.get_parameter_stats(),
    }
    summary_path = assets_dir / f"part1_results_{model_cfg.moe_variant}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Results Saved] Summary JSON written to {summary_path}")

    # Optional WandB logging
    if config.get("wandb_log_eval", True):
        wandb_run = init_wandb(
            config,
            job_type="eval_part1",
            run_name=f"eval_{model_cfg.moe_variant}_{config.get('profile', 'full')}",
            tags=["part1", model_cfg.moe_variant, str(config.get("profile", "full"))],
        )
        if wandb_run is not None:
            import wandb
            log_dict = {
                "eval/test_loss": test_loss,
                "eval/test_perplexity": test_ppl,
                **bleu_results,
            }
            if heatmap_path.exists():
                log_dict["eval/expert_heatmap"] = wandb.Image(str(heatmap_path))
            wandb_run.log(log_dict)
            wandb_run.finish()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Part 1 MoE / Dense Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    args = parser.parse_args()
    evaluate_part1(args.config, args.checkpoint)
