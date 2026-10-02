import argparse
import math
import random
import time
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from src.utils import (
    get_device,
    init_wandb,
    load_checkpoint,
    load_checkpoint_metadata,
    load_config,
    push_checkpoint_artifact_to_hub,
    save_checkpoint,
    set_seed,
)
from src.part1.model import TransformerConfig, Transformer
from src.part1.data import build_or_load_translation_tokenizer
from src.part2.optimizers import (
    CustomAdamW,
    Mars,
    MemoryEfficientOptimizer,
    MatrixBasedOptimizer,
    SophiaH,
)
from src.common.trainer import GenericTrainer, warmup_cosine_lr


class PretrainingTextDataset(Dataset):
    """Concatenate and chunk tokenized text without discarding long-document tails."""
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2 for next-token prediction")
        self.max_seq_len = int(max_seq_len)
        self.pad_token_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        self.tokens = array("I")
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            self.tokens.extend(encoded)
            if eos_id is not None:
                self.tokens.append(int(eos_id))
        self.chunk_starts = list(range(0, max(0, len(self.tokens) - 1), self.max_seq_len - 1))
        self.num_target_tokens = max(0, len(self.tokens) - 1)

    def __len__(self) -> int:
        return len(self.chunk_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.chunk_starts[idx]
        tokens = list(self.tokens[start:start + self.max_seq_len])
        if len(tokens) < self.max_seq_len:
            tokens.extend([self.pad_token_id] * (self.max_seq_len - len(tokens)))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels}


def load_pretraining_data(
    dataset_name: str = "browndw/human-ai-parallel-corpus",
    max_samples: Optional[int] = 50000,
    seed: int = 42,
    allow_synthetic: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Load text corpus and split into train, val, and test partitions."""
    print(f"[Part 2 Data] Loading pretraining corpus '{dataset_name}'...")
    try:
        split_spec = f"train[:{max_samples}]" if max_samples is not None else "train"
        ds = load_dataset(dataset_name, split=split_spec)
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(
                f"Could not load pretraining dataset '{dataset_name}'. Check its Hub ID, "
                "network access, and permissions."
            ) from e
        print(f"[Part 2 Data Warning] Using synthetic smoke text because '{dataset_name}' is unavailable: {e}")
        synthetic = [
            {"text": "Artificial intelligence continues to transform modern NLP architectures. <eos> Recent advances improve language modeling."},
            {"text": "Decoupled weight decay can improve generalization. <eos> Optimizers affect transformer convergence."},
            {"text": "Memory efficient methods reduce optimizer state. <eos> Matrix methods precondition parameter updates."},
        ] * max(10, (max_samples or 300) // 3)
        ds = HFDataset.from_list(synthetic)

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    texts: List[str] = []
    col_names = ds.column_names
    content_columns = [
        column for column in ("text", "human", "ai", "content", "prompt", "response")
        if column in col_names
    ]
    if not content_columns:
        content_columns = [column for column in col_names if column.lower() not in {"id", "doc_id", "index"}]
    for row in ds:
        row_text_parts = [str(row[c]).strip() for c in content_columns if row[c] is not None]
        combined = " <eos> ".join([p for p in row_text_parts if p])
        if len(combined) > 20:
            texts.append(combined)

    random.Random(seed).shuffle(texts)
    n_total = len(texts)
    n_val = max(1, round(n_total * 0.01)) if n_total >= 3 else 0
    n_test = max(1, round(n_total * 0.01)) if n_total >= 3 else 0
    n_train = max(1, n_total - n_val - n_test) if n_total else 0

    train_texts = texts[:n_train]
    val_texts = texts[n_train:n_train + n_val]
    test_texts = texts[n_train + n_val:n_train + n_val + n_test]

    print(f"[Part 2 Data] Pretraining texts: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    return train_texts, val_texts, test_texts


def get_optimizer_instance(
    optimizer_name: str,
    model: nn.Module,
    config: Dict[str, Any],
) -> Any:
    """Instantiate the designated from-scratch optimizer class."""
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.01))
    opt_name = optimizer_name.lower().replace("-", "_")

    if opt_name == "adamw":
        print(f"[Optimizer] Initializing CustomAdamW (lr={lr}, weight_decay={weight_decay})")
        return CustomAdamW(
            model.parameters(),
            lr=lr,
            betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif opt_name in ["mars", "variance_reduced_adamw", "variance_reduced"]:
        print(f"[Optimizer] Initializing Mars (lr={lr}, weight_decay={weight_decay})")
        return Mars(
            model.parameters(),
            lr=lr,
            betas=(float(config.get("beta1", 0.98)), float(config.get("beta2", 0.99))),
            gamma=float(config.get("gamma", 0.05)),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=weight_decay,
            grad_norm=float(config.get("optimizer_grad_norm", 1.0)),
        )
    elif opt_name in ["memory_efficient", "lion"]:
        print(f"[Optimizer] Initializing MemoryEfficientOptimizer / Lion (lr={lr}, weight_decay={weight_decay})")
        return MemoryEfficientOptimizer(
            model.parameters(),
            lr=lr,
            betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.99))),
            weight_decay=weight_decay,
            # GenericTrainer applies the same global clip to every optimizer;
            # disable the optimizer's optional second global clip here.
            grad_norm=0.0,
        )
    elif opt_name in ["matrix_based", "muon"]:
        print(f"[Optimizer] Initializing MatrixBasedOptimizer / Muon (lr={lr}, weight_decay={weight_decay})")
        matrix_params = []
        fallback_params = []
        for name, parameter in model.named_parameters():
            is_embedding = "tok_embeddings" in name or "pos_embeddings" in name
            if parameter.ndim >= 2 and not is_embedding:
                matrix_params.append(parameter)
            else:
                fallback_params.append(parameter)
        return MatrixBasedOptimizer(
            [
                {"params": matrix_params, "use_matrix": True},
                {"params": fallback_params, "use_matrix": False},
            ],
            lr=lr,
            momentum=float(config.get("momentum", 0.95)),
            weight_decay=weight_decay,
            ns_steps=int(config.get("ns_steps", 5)),
            # GenericTrainer applies the shared global clip to every optimizer.
            grad_norm=0.0,
        )
    elif opt_name in ["sophia", "sophiah", "bonus"]:
        print(f"[Optimizer] Initializing SophiaH (Bonus) (lr={lr}, weight_decay={weight_decay})")
        return SophiaH(
            model.parameters(),
            lr=lr,
            betas=(float(config.get("beta1", 0.96)), float(config.get("beta2", 0.99))),
            gamma=float(config.get("gamma", 0.04)),
            hessian_update_interval=int(config.get("hessian_update_interval", 10)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: '{optimizer_name}'. Must be one of: adamw, mars, memory_efficient, matrix_based, sophia")


def evaluate_pretraining_val(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 40,
) -> Tuple[float, float]:
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss = 0.0
    valid_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            count = int((labels[:, 1:] != -100).sum().item())
            total_loss += float(out["lm_loss"].item()) * count
            valid_tokens += count

    avg_loss = total_loss / max(1, valid_tokens)
    ppl = math.exp(min(avg_loss, 50.0))
    return avg_loss, ppl


def _trim_lm_batch(
    batch: Dict[str, torch.Tensor],
    target_tokens: int,
    pad_token_id: int,
) -> Tuple[Optional[Dict[str, torch.Tensor]], int]:
    """Trim to an exact number of non-padding next-token targets."""
    selected_inputs = []
    selected_labels = []
    remaining = target_tokens
    for row_index in range(batch["input_ids"].size(0)):
        input_row = batch["input_ids"][row_index]
        label_row = batch["labels"][row_index]
        valid_length = int((input_row != pad_token_id).sum().item())
        available_targets = max(0, valid_length - 1)
        take_targets = min(available_targets, remaining)
        if take_targets:
            sequence_length = take_targets + 1
            selected_inputs.append(input_row[:sequence_length])
            selected_labels.append(label_row[:sequence_length])
            remaining -= take_targets
        if remaining == 0:
            break
    consumed = target_tokens - remaining
    if not selected_inputs:
        return None, 0

    width = max(row.numel() for row in selected_inputs)
    input_ids = torch.stack([
        torch.nn.functional.pad(row, (0, width - row.numel()), value=pad_token_id)
        for row in selected_inputs
    ])
    labels = torch.stack([
        torch.nn.functional.pad(row, (0, width - row.numel()), value=-100)
        for row in selected_labels
    ])
    return {"input_ids": input_ids, "labels": labels}, consumed


def train_part2(
    config_path: str,
    profile: Optional[str] = None,
    sophia: bool = False,
    resume_from: Optional[str] = None,
):
    config = load_config(config_path)
    if sophia:
        config["optimizer_name"] = "sophia"
    if profile is not None:
        config["profile"] = profile
    profile = str(config.get("profile", "full")).lower()
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be either 'smoke' or 'full'")
    if profile == "smoke":
        config["d_model"] = int(config.get("smoke_d_model", 64))
        config["n_layers"] = int(config.get("smoke_n_layers", 2))
        config["n_heads"] = int(config.get("smoke_n_heads", 4))
        config["d_ff"] = int(config.get("smoke_d_ff", 256))
        config["vocab_size"] = int(config.get("smoke_vocab_size", 512))
        config["max_seq_len"] = int(config.get("smoke_max_seq_len", 64))
        config["batch_size"] = int(config.get("smoke_batch_size", 4))
        config["max_samples"] = int(config.get("smoke_max_samples", 128))
        config["tokenizer_train_samples"] = int(config.get("smoke_tokenizer_train_samples", 64))
        config["allow_synthetic_data"] = bool(config.get("smoke_allow_synthetic_data", True))
    set_seed(config.get("seed", 42))

    device = get_device()
    print(f"[Part 2 Train] Device: {device}")

    # 1. Tokenizer
    tokenizer_dir = config.get("tokenizer_dir", "./data/tokenizer_part1")
    tokenizer_config = {
        **config,
        "dataset_name": config.get(
            "tokenizer_dataset_name", "belumind/en-vi-ja-curated-500k-triplets"
        ),
    }
    tokenizer = build_or_load_translation_tokenizer(
        config=tokenizer_config,
        tokenizer_dir=tokenizer_dir,
        num_train_samples=int(config.get("tokenizer_train_samples", 50000)),
    )

    # 2. Data
    dataset_name = config.get("dataset_name", "browndw/human-ai-parallel-corpus")
    max_samples = config.get("max_samples", 50000)
    train_texts, val_texts, test_texts = load_pretraining_data(
        dataset_name,
        max_samples=max_samples,
        seed=int(config.get("seed", 42)),
        allow_synthetic=bool(config.get("allow_synthetic_data", False)),
    )

    max_seq_len = config.get("max_seq_len", 512)
    batch_size = config.get("batch_size", 16)

    train_ds = PretrainingTextDataset(train_texts, tokenizer, max_seq_len=max_seq_len)
    val_ds = PretrainingTextDataset(val_texts, tokenizer, max_seq_len=max_seq_len)

    train_generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        num_workers=int(config.get("num_workers", 0)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model: Dense-baseline decoder-only transformer (Part 1 Variant 1) trained FRESH
    model_cfg = TransformerConfig(
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 8),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        vocab_size=len(tokenizer),
        max_seq_len=max_seq_len,
        dropout=config.get("dropout", 0.1),
        use_moe=False,
        moe_variant="dense",
    )
    model = Transformer(model_cfg).to(device)

    # 4. Optimizer from scratch
    optimizer_name = config.get("optimizer_name", "adamw")
    optimizer = get_optimizer_instance(optimizer_name, model, config)
    resume_from = resume_from or config.get("resume_from")
    resume_state = None
    if resume_from:
        resume_state = load_checkpoint(resume_from, model, optimizer, device=str(device))
        if resume_state.get("config", {}).get("profile") != profile:
            raise ValueError("Resume checkpoint profile does not match the requested profile")
    before_backward = optimizer.update_hessian if isinstance(optimizer, SophiaH) else None
    trainer = GenericTrainer(
        model,
        optimizer,
        device,
        grad_clip=float(config.get("grad_clip", 1.0)),
        before_backward=before_backward,
    )

    # 5. Token Budget Calculations
    budget_setting = config.get("token_budget", "dataset_1x")
    if budget_setting in (None, "dataset_1x"):
        target_token_budget = train_ds.num_target_tokens
        if target_token_budget < 1:
            raise ValueError("The tokenized Part 2 training split has no next-token targets")
        config["effective_train_tokens_1x"] = target_token_budget
    else:
        target_token_budget = int(budget_setting)
    if profile == "smoke":
        target_token_budget = int(config.get("smoke_token_budget", 8192))
    if resume_state:
        saved_budget = resume_state.get("config", {}).get("effective_train_token_budget")
        if saved_budget != target_token_budget:
            raise ValueError(
                f"Resume checkpoint budget ({saved_budget}) does not match configured budget ({target_token_budget})"
            )
    nominal_tokens_per_step = batch_size * max(1, max_seq_len - 1)
    total_steps = max(1, math.ceil(target_token_budget / nominal_tokens_per_step))
    milestones = [math.ceil(target_token_budget * i / 10) for i in range(1, 11)]
    warmup_steps = int(config.get("warmup_steps", 2000))
    if profile == "smoke":
        warmup_steps = min(warmup_steps, max(0, total_steps // 10))
    base_lr = float(config.get("lr", 3e-4))
    min_lr_ratio = float(config.get("minimum_lr_ratio", 0.1))

    def learning_rate(step: int) -> float:
        return warmup_cosine_lr(step, base_lr, total_steps, warmup_steps, min_lr_ratio)

    print("\n" + "=" * 60)
    print(f"Optimizer:               {optimizer_name.upper()}")
    print(f"Effective 1x Training Targets: {target_token_budget:,} non-padding tokens")
    print(f"Nominal Tokens Per Step: {nominal_tokens_per_step:,} (batch_size={batch_size}, seq_len={max_seq_len})")
    print(f"Total Steps to 1.0x:     {total_steps:,} steps")
    print("0.1x Token Milestones:   " + ", ".join(f"{index / 10:.1f}x: {tokens:,}" for index, tokens in enumerate(milestones, 1)))
    print("=" * 60 + "\n")

    # 6. WandB Init
    wandb_run = init_wandb(
        config=config,
        job_type="train_part2",
        run_name=config.get("run_name", f"part2_{optimizer_name}"),
        tags=["part2", optimizer_name, profile],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if wandb_run is not None:
        wandb_run.config.update({
            "model/total_params": parameter_count,
            "model/active_params": parameter_count,
            "train/token_budget": target_token_budget,
            "train/batch_size": int(batch_size),
            "train/sequence_length": int(max_seq_len),
            "optimizer/name": optimizer_name,
        }, allow_val_change=True)

    save_dir = Path(config.get("checkpoint_dir", f"./runs/checkpoints/part2_{optimizer_name}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 7. Training Loop
    global_step = int(resume_state.get("step", 0)) if resume_state else 0
    tokens_seen = int(resume_state.get("config", {}).get("tokens_seen", 0)) if resume_state else 0
    resume_checkpoint_path = (
        Path(resume_from).with_suffix(".safetensors") if resume_from else None
    )
    if tokens_seen > target_token_budget:
        raise ValueError("Resume checkpoint has already exceeded the configured token budget")
    periodic_checkpoints = []
    if (
        resume_checkpoint_path is not None
        and resume_checkpoint_path.parent.resolve() == save_dir.resolve()
    ):
        periodic_checkpoints.append(resume_checkpoint_path)
    running_loss = 0.0
    running_tokens = 0
    milestone_index = (
        min(10, int(round(tokens_seen / target_token_budget * 10)))
        if tokens_seen else 0
    )
    pbar = tqdm(
        total=target_token_budget,
        initial=tokens_seen,
        desc=f"Part 2 ({optimizer_name}) tokens",
    )

    model.train()
    train_iterator = iter(train_loader)
    for _ in range(global_step):
        try:
            next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            next(train_iterator)
    last_progress = time.perf_counter()
    while tokens_seen < target_token_budget:
        if milestone_index >= len(milestones):
            break
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        next_milestone = milestones[milestone_index]
        target_count = min(next_milestone - tokens_seen, target_token_budget - tokens_seen)
        trimmed, consumed = _trim_lm_batch(batch, target_count, tokenizer.pad_token_id or 0)
        if trimmed is None or consumed == 0:
            continue
        input_ids = trimmed["input_ids"].to(device)
        labels = trimmed["labels"].to(device)

        current_lr = learning_rate(global_step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        step_metrics = trainer.train_batch(input_ids, labels)
        loss_value = step_metrics["lm_loss"]

        global_step += 1
        tokens_seen += consumed
        running_loss += loss_value * consumed
        running_tokens += consumed
        pbar.update(consumed)
        pbar.set_postfix({"loss": f"{loss_value:.4f}", "tokens": f"{tokens_seen:,}"})

        if global_step % max(1, int(config.get("log_interval", 20))) == 0:
            elapsed = max(1e-9, time.perf_counter() - last_progress)
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": running_loss / max(1, running_tokens),
                    "train/learning_rate": current_lr,
                    "train/grad_norm": step_metrics["grad_norm"],
                    "train/tokens_seen": tokens_seen,
                    "train/tokens_per_second": running_tokens / elapsed,
                }, step=global_step)
            running_loss = 0.0
            running_tokens = 0
            last_progress = time.perf_counter()

        if tokens_seen >= next_milestone:
            fraction = round((milestone_index + 1) / 10, 1)
            val_loss, val_ppl = evaluate_pretraining_val(
                model, val_loader, device,
                max_batches=int(config.get("max_val_batches", 40)),
            )
            print(
                f"\n[Milestone {fraction:.1f}x | Step {global_step} | Tokens {tokens_seen:,}] "
                f"Validation loss: {val_loss:.4f} | Validation perplexity: {val_ppl:.3f}"
            )
            milestone_config = {
                **config,
                "profile": profile,
                "effective_train_token_budget": target_token_budget,
                "tokens_seen": tokens_seen,
                "fraction": fraction,
                "evaluation_data": "validation split of browndw/human-ai-parallel-corpus; no BLEU references available",
                "validation_perplexity": val_ppl,
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
            save_checkpoint(
                model=model,
                optimizer=None,
                step=global_step,
                epoch=0,
                loss=val_loss,
                config=milestone_config,
                save_dir=save_dir,
                filename=f"checkpoint_{fraction:.1f}x.pt",
            )
            if wandb_run is not None:
                wandb_run.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "train/tokens_seen": tokens_seen,
                    "train/milestone_fraction": fraction,
                }, step=global_step)
            model.train()
            milestone_index += 1

        checkpoint_interval = max(1, int(config.get("checkpoint_interval_steps", 1000)))
        if global_step % checkpoint_interval == 0 and tokens_seen < target_token_budget:
            resume_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=global_step,
                epoch=0,
                loss=loss_value,
                config={
                    **config,
                    "profile": profile,
                    "effective_train_token_budget": target_token_budget,
                    "tokens_seen": tokens_seen,
                    "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                },
                save_dir=save_dir,
                filename=f"checkpoint_step_{global_step}.pt",
            )
            periodic_checkpoints.append(resume_checkpoint_path)
            while len(periodic_checkpoints) > 2:
                old_checkpoint = periodic_checkpoints.pop(0)
                for old_path in (
                    old_checkpoint,
                    old_checkpoint.with_suffix(".json"),
                    old_checkpoint.with_name(f"{old_checkpoint.stem}.optimizer.pt"),
                ):
                    old_path.unlink(missing_ok=True)

    pbar.close()
    if tokens_seen != target_token_budget:
        raise RuntimeError(
            f"Part 2 token budget mismatch: expected {target_token_budget}, consumed {tokens_seen}"
        )
    print(f"[Part 2 Token Budget] consumed exactly {tokens_seen:,} non-padding next-token targets.")
    print(f"\n[Part 2 Completed] Finished training {optimizer_name} up to 1.0x token budget ({tokens_seen:,} target tokens).")
    if profile == "full" and config.get("push_to_hub", False) and config.get("hf_repo_id"):
        checkpoint_path = save_dir / "checkpoint_1.0x.safetensors"
        metadata = load_checkpoint_metadata(checkpoint_path)
        if not push_checkpoint_artifact_to_hub(
            checkpoint_path,
            str(config["hf_repo_id"]),
            metadata,
            tokenizer_dir=tokenizer_dir,
            commit_message=f"Upload final Part 2 {optimizer_name} checkpoint",
        ):
            raise RuntimeError("Hugging Face upload failed; verify login and repository permissions.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Part 2 Optimizers")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--sophia", action="store_true", help="Use the optional Sophia-H optimizer")
    parser.add_argument("--profile", choices=("smoke", "full"), default=None)
    parser.add_argument("--resume-from", default=None, help="Resume model and optimizer state from a checkpoint")
    args = parser.parse_args()
    train_part2(args.config, profile=args.profile, sophia=args.sophia, resume_from=args.resume_from)
