import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from src.part1.data import build_or_load_translation_tokenizer, get_translation_dataloaders
from src.part1.model import Transformer, TransformerConfig
from src.part2.optimizers import CustomAdamW
from src.common.trainer import GenericTrainer, warmup_cosine_lr
from src.utils import (
	get_device,
	init_wandb,
	load_config,
	load_checkpoint_metadata,
	load_checkpoint,
	push_checkpoint_artifact_to_hub,
	save_checkpoint,
	set_seed,
)


def _validate(model: Transformer, loader, device: torch.device, max_batches: int) -> Dict[str, float]:
	model.eval()
	loss_sum = 0.0
	token_count = 0
	with torch.no_grad():
		for batch_index, batch in enumerate(loader):
			if batch_index >= max_batches:
				break
			input_ids = batch["input_ids"].to(device)
			labels = batch["labels"].to(device)
			output = model(
				input_ids,
				labels=labels,
				attention_mask=batch["attention_mask"].to(device),
			)
			count = int((labels[:, 1:] != -100).sum().item())
			loss_sum += float(output["lm_loss"].item()) * count
			token_count += count
	average_loss = loss_sum / max(1, token_count)
	return {"loss": average_loss, "perplexity": math.exp(min(average_loss, 50.0))}


def _limit_batch_tokens(batch: Dict[str, Any], remaining_tokens: int, pad_id: int) -> Dict[str, Any]:
	"""Trim by real input tokens while guaranteeing each retained row has a target."""
	selected = []
	remaining = remaining_tokens
	for row in range(batch["input_ids"].size(0)):
		length = int(batch["attention_mask"][row].sum().item())
		row_labels = batch["labels"][row]
		valid_targets = torch.nonzero(row_labels[1:length] != -100, as_tuple=False).flatten()
		if valid_targets.numel() == 0:
			continue
		minimum_length = int(valid_targets[0].item()) + 2
		if remaining < minimum_length:
			continue
		take = min(length, remaining)
		selected.append((batch["input_ids"][row, :take], row_labels[:take]))
		remaining -= take
		if remaining == 0:
			break

	if not selected:
		return {
			"input_ids": batch["input_ids"][:0],
			"labels": batch["labels"][:0],
			"attention_mask": batch["attention_mask"][:0],
		}

	width = max(row[0].numel() for row in selected)
	input_rows = []
	label_rows = []
	attention_rows = []
	for input_row, label_row in selected:
		padding = width - input_row.numel()
		input_rows.append(torch.nn.functional.pad(input_row, (0, padding), value=pad_id))
		label_rows.append(torch.nn.functional.pad(label_row, (0, padding), value=-100))
		attention_rows.append(torch.nn.functional.pad(torch.ones_like(input_row), (0, padding), value=0))
	return {
		"input_ids": torch.stack(input_rows),
		"labels": torch.stack(label_rows),
		"attention_mask": torch.stack(attention_rows),
	}


def train_part1(
	config_path: str,
	profile: Optional[str] = None,
	resume_from: Optional[str] = None,
) -> Dict[str, Any]:
	config = load_config(config_path)
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
		config["max_seq_len"] = int(config.get("smoke_max_seq_len", 64))
		config["vocab_size"] = int(config.get("smoke_vocab_size", 512))
		config["max_samples"] = int(config.get("smoke_max_samples", 64))
		config["tokenizer_train_samples"] = int(config.get("smoke_tokenizer_train_samples", 64))
		config["allow_synthetic_data"] = bool(config.get("smoke_allow_synthetic_data", True))

	set_seed(int(config.get("seed", 42)))
	device = get_device()
	print(f"[Part 1 Train] profile={profile}, device={device}")

	tokenizer = build_or_load_translation_tokenizer(
		config=config,
		tokenizer_dir=config.get("tokenizer_dir", "./data/tokenizer_part1"),
		num_train_samples=int(config.get("tokenizer_train_samples", 50000)),
	)
	train_loader, val_loader, _ = get_translation_dataloaders(config, tokenizer)

	model_config = TransformerConfig(
		d_model=int(config.get("d_model", 512)),
		n_layers=int(config.get("n_layers", 8)),
		n_heads=int(config.get("n_heads", 8)),
		d_ff=int(config.get("d_ff", 2048)),
		vocab_size=len(tokenizer),
		max_seq_len=int(config.get("max_seq_len", 256)),
		dropout=float(config.get("dropout", 0.1)),
		use_moe=bool(config.get("use_moe", False)),
		moe_variant=str(config.get("moe_variant", "dense")),
		n_experts=int(config.get("n_experts", 4)),
		top_k=int(config.get("top_k", 1)),
		aux_loss_coef=float(config.get("aux_loss_coef", 0.01)),
		router_z_loss_coef=float(config.get("router_z_loss_coef", 1e-3)),
	)
	model = Transformer(model_config).to(device)
	optimizer = CustomAdamW(
		model.parameters(),
		lr=float(config.get("lr", 3e-4)),
		betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
		weight_decay=float(config.get("weight_decay", 0.01)),
	)
	resume_from = resume_from or config.get("resume_from")
	resume_state = None
	if resume_from:
		resume_state = load_checkpoint(resume_from, model, optimizer, device=str(device))
		if resume_state.get("config", {}).get("profile") != profile:
			raise ValueError("Resume checkpoint profile does not match the requested profile")
	trainer = GenericTrainer(
		model,
		optimizer,
		device,
		grad_clip=float(config.get("grad_clip", 1.0)),
	)

	full_budget = int(config.get("token_budget", 30_000_000))
	token_budget = int(config.get("smoke_token_budget", 8192)) if profile == "smoke" else full_budget
	if token_budget < 1:
		raise ValueError("token budget must be positive")
	if resume_state:
		saved_budget = resume_state.get("config", {}).get("training_token_budget")
		if saved_budget != token_budget:
			raise ValueError(
				f"Resume checkpoint budget ({saved_budget}) does not match configured budget ({token_budget})"
			)
	total_steps = max(1, math.ceil(token_budget / max(1, int(config.get("tokens_per_step", 16384)))))
	warmup_steps = int(config.get("warmup_steps", 2000))
	if profile == "smoke":
		warmup_steps = min(warmup_steps, max(0, total_steps // 10))
	minimum_lr_ratio = float(config.get("minimum_lr_ratio", 0.1))
	base_lr = float(config.get("lr", 3e-4))

	def learning_rate(step: int) -> float:
		return warmup_cosine_lr(step, base_lr, total_steps, warmup_steps, minimum_lr_ratio)

	run_name = config.get("run_name", f"part1_{model_config.moe_variant}_{profile}")
	wandb_run = init_wandb(
		config,
		job_type="train_part1",
		run_name=run_name,
		tags=["part1", model_config.moe_variant, profile],
	)
	parameter_stats = model.get_parameter_stats()
	if wandb_run is not None:
		wandb_run.config.update({
			"model/total_params": parameter_stats["total_params"],
			"model/active_params": parameter_stats["active_params"],
			"model/d_expert": parameter_stats["d_expert"],
			"train/token_budget": token_budget,
			"train/batch_size": int(config.get("batch_size", 64)),
			"train/sequence_length": model_config.max_seq_len,
		}, allow_val_change=True)

	checkpoint_dir = Path(config.get(
		"checkpoint_dir", f"./runs/checkpoints/part1_{model_config.moe_variant}"
	))
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	periodic_checkpoints = []
	if resume_from:
		resume_checkpoint_path = Path(resume_from).with_suffix(".safetensors")
		if resume_checkpoint_path.parent.resolve() == checkpoint_dir.resolve():
			periodic_checkpoints.append(resume_checkpoint_path)
	best_loss = float(resume_state.get("loss", float("inf"))) if resume_state else float("inf")
	tokens_seen = int(resume_state.get("config", {}).get("tokens_seen", 0)) if resume_state else 0
	step = int(resume_state.get("step", 0)) if resume_state else 0
	started = time.perf_counter()
	train_iterator = iter(train_loader)
	for _ in range(step):
		try:
			next(train_iterator)
		except StopIteration:
			train_iterator = iter(train_loader)
			next(train_iterator)
	model.train()

	progress = tqdm(total=token_budget, desc=f"Part 1 {model_config.moe_variant} tokens")
	unfit_batches = 0
	while tokens_seen < token_budget:
		try:
			batch = next(train_iterator)
		except StopIteration:
			train_iterator = iter(train_loader)
			batch = next(train_iterator)
		remaining = token_budget - tokens_seen
		trimmed = _limit_batch_tokens(batch, remaining, tokenizer.pad_token_id or 0)
		if trimmed["input_ids"].numel() == 0:
			if remaining > model_config.max_seq_len:
				raise RuntimeError(
					"No target-bearing training example fit despite more than one context window remaining."
				)
			unfit_batches = max(1, len(train_loader))
			print("[Part 1 Token Budget] No complete target-bearing example fits the remaining short tail; stopping cleanly.")
			break
		unfit_batches = 0
		input_ids = trimmed["input_ids"].to(device)
		labels = trimmed["labels"].to(device)
		attention_mask = trimmed["attention_mask"].to(device)
		processed_tokens = int(attention_mask.sum().item())
		valid_targets = int((labels[:, 1:] != -100).sum().item())
		if valid_targets == 0:
			progress.close()
			print("[Part 1 Token Budget] Remaining budget cannot fit a complete source-plus-target example; stopping cleanly.")
			break

		current_lr = learning_rate(step)
		for group in optimizer.param_groups:
			group["lr"] = current_lr
		metrics = trainer.train_batch(input_ids, labels, attention_mask=attention_mask)

		tokens_seen += processed_tokens
		step += 1
		progress.update(processed_tokens)
		progress.set_postfix({"loss": f"{metrics['lm_loss']:.3f}", "lr": f"{current_lr:.2e}"})

		log_interval = max(1, int(config.get("log_interval", 10)))
		if wandb_run is not None and step % log_interval == 0:
			wandb_run.log({
				"train/loss": metrics["loss"],
				"train/lm_loss": metrics["lm_loss"],
				"train/load_balance_loss": metrics["aux_loss"],
				"train/router_z_loss": metrics["router_z_loss"],
				"train/learning_rate": current_lr,
				"train/grad_norm": metrics["grad_norm"],
				"train/tokens_seen": tokens_seen,
				"train/tokens_per_second": tokens_seen / max(1e-9, time.perf_counter() - started),
			}, step=step)

		eval_interval = max(1, int(config.get("eval_interval_steps", 500)))
		if step % eval_interval == 0 or tokens_seen >= token_budget:
			metrics = _validate(
				model,
				val_loader,
				device,
				max_batches=int(config.get("max_val_batches", 20)),
			)
			print(f"\n[Part 1 validation] step={step} tokens={tokens_seen} loss={metrics['loss']:.4f}")
			checkpoint_interval = max(1, int(config.get("checkpoint_interval_steps", 1000)))
			if tokens_seen < token_budget and step % checkpoint_interval == 0:
				checkpoint_path = save_checkpoint(
					model, optimizer, step, 0, metrics["loss"],
					{
						**config,
						"profile": profile,
						"tokens_seen": tokens_seen,
						"training_token_budget": token_budget,
						"model_parameter_stats": parameter_stats,
					},
					checkpoint_dir,
					filename=f"checkpoint_step_{step}.pt",
				)
				periodic_checkpoints.append(checkpoint_path)
				while len(periodic_checkpoints) > 2:
					old_checkpoint = periodic_checkpoints.pop(0)
					old_paths = (
						old_checkpoint,
						old_checkpoint.with_suffix(".json"),
						old_checkpoint.with_name(f"{old_checkpoint.stem}.optimizer.pt"),
					)
					for old_path in old_paths:
						old_path.unlink(missing_ok=True)
			if metrics["loss"] < best_loss:
				best_loss = metrics["loss"]
				save_checkpoint(
					model, None, step, 0, best_loss,
					{
						**config,
						"profile": profile,
						"tokens_seen": tokens_seen,
						"training_token_budget": token_budget,
					"model_parameter_stats": parameter_stats,
					},
					checkpoint_dir,
					filename="best_model.pt",
				)
				(checkpoint_dir / "best_model.optimizer.pt").unlink(missing_ok=True)
			if wandb_run is not None:
				wandb_run.log({
					"val/loss": metrics["loss"],
					"val/perplexity": metrics["perplexity"],
					"val/tokens_seen": tokens_seen,
				}, step=step)
			model.train()

	progress.close()
	budget_shortfall = token_budget - tokens_seen
	clean_tail_stop = (
		tokens_seen > 0
		and 0 < budget_shortfall <= model_config.max_seq_len
		and unfit_batches >= max(1, len(train_loader))
	)
	if tokens_seen < token_budget:
		print(f"[Part 1 Token Budget] Stopped at {tokens_seen:,}/{token_budget:,} input tokens because no complete target-bearing example fit the remainder.")
	else:
		print(f"[Part 1 Token Budget] {model_config.moe_variant} consumed exactly {tokens_seen:,} tokens ({profile}).")
	if tokens_seen > 0:
		save_checkpoint(
			model, optimizer, step, 0, best_loss,
			{
				**config,
				"profile": profile,
				"tokens_seen": tokens_seen,
				"training_token_budget": token_budget,
				"budget_stopped_early": tokens_seen < token_budget,
				"budget_shortfall_tokens": budget_shortfall,
				"model_parameter_stats": parameter_stats,
			},
			checkpoint_dir,
			filename="final_model.pt",
		)
	if (
		profile == "full"
		and (tokens_seen == token_budget or clean_tail_stop)
		and config.get("push_to_hub", False)
		and config.get("hf_repo_id")
	):
		checkpoint_path = checkpoint_dir / "final_model.safetensors"
		metadata = load_checkpoint_metadata(checkpoint_path)
		if not push_checkpoint_artifact_to_hub(
			checkpoint_path,
			str(config["hf_repo_id"]),
			metadata,
			tokenizer_dir=config.get("tokenizer_dir", "./data/tokenizer_part1"),
			commit_message=f"Upload final Part 1 {model_config.moe_variant} checkpoint",
		):
			raise RuntimeError("Hugging Face upload failed; verify login and repository permissions.")
	elif profile == "full" and config.get("push_to_hub", False) and tokens_seen < token_budget:
		print("[Hugging Face] Skipping final upload because training stopped before the allowed clean tail remainder.")
	if wandb_run is not None:
		wandb_run.finish()
	return {
		"variant": model_config.moe_variant,
		"profile": profile,
		"tokens_seen": tokens_seen,
		"steps": step,
		"best_validation_loss": best_loss,
		"checkpoint_dir": str(checkpoint_dir),
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a Part 1 multilingual translation model")
	parser.add_argument("--config", required=True, help="YAML experiment configuration")
	parser.add_argument("--profile", choices=("smoke", "full"), default=None)
	parser.add_argument("--resume-from", default=None, help="Resume model and optimizer state from a checkpoint")
	args = parser.parse_args()
	train_part1(args.config, profile=args.profile, resume_from=args.resume_from)
