import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.utils import load_checkpoint
from src.part1.model import TransformerConfig, Transformer
from src.utils import init_wandb, load_checkpoint_metadata, load_config, load_multilingual_tokenizer, get_device
from src.common.decoding import greedy_decode


def compute_test_bleu(
    model: Transformer,
    tokenizer,
    test_texts: List[str],
    device: torch.device,
    num_samples: int = 100,
    max_new_tokens: int = 40,
) -> float:
    """Generative test BLEU: greedy-decode each test document's second half from its first half
    (via model.forward() only) and score corpus BLEU against the held-out reference continuation."""
    import sacrebleu

    model.eval()
    eos_id = tokenizer.eos_token_id
    hyps, refs = [], []
    for text in test_texts[:num_samples]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        split = max(1, len(ids) // 2)
        prompt_ids, ref_ids = ids[:split], ids[split:]
        if not ref_ids:
            continue
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = greedy_decode(
            model, prompt, max_new_tokens=min(max_new_tokens, len(ref_ids)),
            eos_token_id=eos_id, pad_token_id=eos_id,
        )
        generated = output[0, len(prompt_ids):].tolist()
        hyps.append(tokenizer.decode(generated, skip_special_tokens=True))
        refs.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
    return sacrebleu.corpus_bleu(hyps, [refs]).score if hyps else 0.0
from src.part2.train import (
    PretrainingTextDataset,
    load_pretraining_data,
    evaluate_pretraining_val,
)


def evaluate_single_checkpoint(
    checkpoint_path: Path,
    model: Transformer,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    tokenizer=None,
    test_texts: List[str] = None,
    bleu_samples: int = 100,
    bleu_max_new_tokens: int = 40,
) -> Dict[str, Any]:
    """Load a checkpoint and report token-weighted validation and held-out test loss."""
    state = load_checkpoint(checkpoint_path, model=model, device=str(device))
    config = state.get("config", {})
    tokens_seen = config.get("tokens_seen", state.get("step", 0) * 16 * 512)
    fraction = config.get("fraction", 0.0)

    val_loss, val_ppl = evaluate_pretraining_val(model, val_loader, device)

    test_loss, test_ppl = evaluate_pretraining_val(model, test_loader, device)

    test_bleu = None
    if tokenizer is not None and test_texts:
        test_bleu = compute_test_bleu(
            model, tokenizer, test_texts, device,
            num_samples=bleu_samples, max_new_tokens=bleu_max_new_tokens,
        )

    return {
        "fraction": fraction,
        "tokens_seen": tokens_seen,
        "model_parameters": int(config.get("model_parameters", sum(p.numel() for p in model.parameters()))),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "test_bleu": test_bleu,
    }


def evaluate_all_optimizers(
    config_dir: str = "configs/part2",
    output_plot_path: str = "assets/part2_optimizer_comparison_v2.png",
    output_json_path: str = "assets/part2_optimizer_results_v2.json",
    output_csv_path: str = "assets/part2_optimizer_results_v2.csv",
    include_sophia: bool = False,
):
    """
    Evaluates checkpoints for all 4 optimizers at every 0.1x token milestone,
    computes validation loss and held-out test perplexity, then generates a unified comparison chart.
    """
    device = get_device()
    print(f"[Part 2 Evaluation] Evaluating all optimizers on {device}...")

    tokenizer = load_multilingual_tokenizer("./data/tokenizer_part1")

    optimizer_names = [
        "adamw",
        "mars",
        "memory_efficient",
        "matrix_based",
    ]
    if include_sophia:
        optimizer_names.append("sophia")

    results_by_optimizer: Dict[str, List[Dict[str, Any]]] = {}
    summary_rows = []

    for opt_name in optimizer_names:
        ckpt_dir = Path(f"./runs/checkpoints/part2_{opt_name}")
        if not ckpt_dir.exists():
            print(f"[Notice] Checkpoint directory not found for {opt_name}: {ckpt_dir}. Skipping.")
            continue

        print("\n" + "=" * 50)
        print(f"Evaluating Optimizer Milestones: {opt_name.upper()}")
        print("=" * 50)

        opt_results = []
        fractions = [round(0.1 * i, 1) for i in range(1, 11)]
        first_checkpoint = next(
            (candidate for fraction in fractions for candidate in (
                ckpt_dir / f"checkpoint_{fraction:.1f}x.safetensors",
                ckpt_dir / f"checkpoint_{fraction:.1f}x.pt",
            ) if candidate.exists()),
            None,
        )
        if first_checkpoint is None:
            print(f"[Notice] No milestone checkpoints found in {ckpt_dir}; skipping.")
            continue
        first_state = load_checkpoint_metadata(first_checkpoint)
        saved_config = first_state.get("config", {})
        max_seq_len = int(saved_config.get("max_seq_len", 512))
        train_texts, val_texts, test_texts = load_pretraining_data(
            dataset_name=saved_config.get("dataset_name", "browndw/human-ai-parallel-corpus"),
            max_samples=saved_config.get("max_samples", 50000),
            seed=int(saved_config.get("seed", 42)),
            allow_synthetic=bool(saved_config.get("allow_synthetic_data", False)),
        )
        if saved_config.get("profile") == "smoke":
            train_texts, val_texts, test_texts = load_pretraining_data(
                dataset_name=saved_config.get("dataset_name", "browndw/human-ai-parallel-corpus"),
                max_samples=saved_config.get("max_samples", 128),
                seed=int(saved_config.get("seed", 42)),
                allow_synthetic=bool(saved_config.get("allow_synthetic_data", True)),
            )
        val_ds = PretrainingTextDataset(val_texts, tokenizer, max_seq_len=max_seq_len)
        test_ds = PretrainingTextDataset(test_texts, tokenizer, max_seq_len=max_seq_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=int(saved_config.get("batch_size", 16)),
            shuffle=False,
        )
        test_loader = DataLoader(test_ds, batch_size=int(saved_config.get("batch_size", 16)), shuffle=False)
        model_cfg = TransformerConfig(
            d_model=int(saved_config.get("d_model", 512)),
            n_layers=int(saved_config.get("n_layers", 8)),
            n_heads=int(saved_config.get("n_heads", 8)),
            d_ff=int(saved_config.get("d_ff", 2048)),
            vocab_size=len(tokenizer),
            max_seq_len=max_seq_len,
            dropout=0.0,
            use_moe=False,
            moe_variant="dense",
        )
        model = Transformer(model_cfg).to(device)
        run_config_path = Path(config_dir) / f"{opt_name}.yaml"
        run_config = load_config(run_config_path) if run_config_path.exists() else saved_config
        profile = str(saved_config.get("profile", run_config.get("profile", "full")))
        wandb_run = init_wandb(
            run_config,
            job_type="eval_part2",
            run_name=f"eval_part2_{opt_name}_{profile}",
            tags=["part2", opt_name, profile],
        )

        for frac in fractions:
            candidates = [
                ckpt_dir / f"checkpoint_{frac:.1f}x.safetensors",
                ckpt_dir / f"checkpoint_{frac:.1f}x.pt",
            ]
            ckpt_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            if not ckpt_path.exists():
                continue

            print(f"--> Evaluating {ckpt_path.name}...")
            res = evaluate_single_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                tokenizer=tokenizer,
                test_texts=test_texts,
                bleu_samples=int(saved_config.get("bleu_samples", run_config.get("bleu_samples", 100))),
                bleu_max_new_tokens=int(saved_config.get("bleu_max_new_tokens", run_config.get("bleu_max_new_tokens", 40))),
            )
            opt_results.append(res)
            summary_rows.append({"optimizer": opt_name, **res})
            bleu_str = f" | Test BLEU: {res['test_bleu']:.2f}" if res["test_bleu"] is not None else ""
            print(f"    Tokens: {res['tokens_seen']:,} | Val Loss: {res['val_loss']:.4f} | Test PPL: {res['test_ppl']:.3f}{bleu_str}")
            if wandb_run is not None:
                log_payload = {
                    "val/loss": res["val_loss"],
                    "val/perplexity": res["val_ppl"],
                    "eval/test_loss": res["test_loss"],
                    "eval/test_perplexity": res["test_ppl"],
                    "train/tokens_seen": res["tokens_seen"],
                    "train/milestone_fraction": res["fraction"],
                }
                if res["test_bleu"] is not None:
                    log_payload["eval/test_bleu"] = res["test_bleu"]
                wandb_run.log(log_payload, step=int(res.get("tokens_seen", 0)))

        if opt_results:
            opt_results.sort(key=lambda x: x["tokens_seen"])
            results_by_optimizer[opt_name] = opt_results
        if wandb_run is not None:
            wandb_run.finish()

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_by_optimizer, f, indent=2)
    print(f"\n[Results Saved] Optimizer milestone metrics saved to {output_json_path}")

    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["optimizer", "fraction", "tokens_seen", "model_parameters", "chinchilla_ratio", "val_loss", "val_ppl", "test_loss", "test_ppl", "test_bleu"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            row["chinchilla_ratio"] = row["tokens_seen"] / max(1, 20 * row["model_parameters"])
            writer.writerow(row)
    print(f"[Results Saved] Optimizer milestone CSV saved to {csv_path}")

    if results_by_optimizer:
        plot_optimizer_comparison(results_by_optimizer, Path(output_plot_path))
    else:
        print("[Notice] No checkpoints found to plot yet. Run training for Part 2 optimizers first.")


def plot_optimizer_comparison(
    results: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
):
    """Plot validation loss, held-out test perplexity and generative test BLEU against tokens seen."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    colors = {
        "adamw": "#1f77b4",
        "mars": "#ff7f0e",
        "memory_efficient": "#2ca02c",
        "matrix_based": "#d62728",
        "sophia": "#9467bd",
    }
    markers = {
        "adamw": "o",
        "mars": "s",
        "memory_efficient": "^",
        "matrix_based": "D",
        "sophia": "v",
    }
    display_names = {
        "adamw": "AdamW (Decoupled)",
        "mars": "Mars (Variance-Reduced AdamW)",
        "memory_efficient": "Lion (Memory-Efficient)",
        "matrix_based": "Muon (Matrix Newton-Schulz)",
        "sophia": "Sophia-H (2nd-Order Bonus)",
    }

    for opt_name, points in results.items():
        if not points:
            continue
        tokens = [p["tokens_seen"] / 1e6 for p in points]
        val_losses = [p["val_loss"] for p in points]
        test_ppls = [p["test_ppl"] for p in points]

        label = display_names.get(opt_name, opt_name)
        color = colors.get(opt_name, None)
        marker = markers.get(opt_name, "o")

        ax1.plot(tokens, val_losses, label=label, color=color, marker=marker, linewidth=2, markersize=6)
        ax2.plot(tokens, test_ppls, label=label, color=color, marker=marker, linewidth=2, markersize=6)

        bleu_points = [p for p in points if p.get("test_bleu") is not None]
        if bleu_points:
            ax3.plot([p["tokens_seen"] / 1e6 for p in bleu_points], [p["test_bleu"] for p in bleu_points],
                     label=label, color=color, marker=marker, linewidth=2, markersize=6)

    model_parameters = next(
        (int(point["model_parameters"]) for points in results.values() for point in points
         if point.get("model_parameters")),
        1,
    )
    ax1.secondary_xaxis(
        "top",
        functions=(
            lambda tokens_m: tokens_m * 1e6 / (20 * model_parameters),
            lambda ratio: ratio * 20 * model_parameters / 1e6,
        ),
    ).set_xlabel("Chinchilla ratio: tokens / (20 x parameters)")

    ax1.set_title("Validation Loss vs. Tokens Seen (Pretraining)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Tokens Seen (Millions)", fontsize=11)
    ax1.set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_title("Held-out Test Perplexity vs. Tokens Seen", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tokens Seen (Millions)", fontsize=11)
    ax2.set_ylabel("Test Perplexity", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    ax3.set_title("Generative Test BLEU vs. Tokens Seen", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Tokens Seen (Millions)", fontsize=11)
    ax3.set_ylabel("Test BLEU", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot Saved] Optimizer comparison chart saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Part 2 Optimizer Checkpoints")
    parser.add_argument("--config_dir", type=str, default="configs/part2", help="Configs directory")
    parser.add_argument("--output_plot", type=str, default="assets/part2_optimizer_comparison_v2.png")
    parser.add_argument("--output_json", type=str, default="assets/part2_optimizer_results_v2.json")
    parser.add_argument("--output_csv", type=str, default="assets/part2_optimizer_results_v2.csv")
    parser.add_argument("--include-sophia", action="store_true")
    args = parser.parse_args()
    evaluate_all_optimizers(
        args.config_dir, args.output_plot, args.output_json, args.output_csv,
        include_sophia=args.include_sophia,
    )
