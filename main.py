import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from src.part1.evaluate import evaluate_part1
from src.part1.train import train_part1
from src.part2.evaluate import evaluate_all_optimizers
from src.part2.train import train_part2
from src.part3.evaluate import evaluate_part3
from src.utils import load_config
from src.part1.data import build_or_load_translation_tokenizer


PART1_CONFIGS = (
	"configs/part1/dense.yaml",
	"configs/part1/moe_top1.yaml",
	"configs/part1/moe_top2.yaml",
	"configs/part1/moe_shared_routed.yaml",
	"configs/part1/moe_top2_matched_active.yaml",
)
PART2_CONFIGS = (
	"configs/part2/adamw.yaml",
	"configs/part2/variance_reduced_adamw.yaml",
	"configs/part2/memory_efficient.yaml",
	"configs/part2/matrix_based.yaml",
)


def run_part1_all(profile: str) -> List[Dict[str, Any]]:
	rows = []
	for config_path in PART1_CONFIGS:
		print(f"\n=== Part 1 experiment: {config_path} ===")
		training = train_part1(config_path, profile=profile)
		evaluation = evaluate_part1(
			config_path,
			checkpoint_path=str(Path(training["checkpoint_dir"]) / "best_model.safetensors"),
			profile=profile,
		)
		parameter_stats = evaluation["parameter_stats"]
		rows.append({
			"variant": evaluation["variant"],
			"profile": profile,
			"test_loss": evaluation["test_loss"],
			"test_perplexity": evaluation["test_perplexity"],
			"bleu_overall": evaluation["bleu"]["bleu_overall"],
			"bleu_vi_en": evaluation["bleu"]["bleu_vi_en"],
			"bleu_ja_en": evaluation["bleu"]["bleu_ja_en"],
			"total_params": parameter_stats["total_params"],
			"active_params": parameter_stats["active_params"],
		})
	output_path = Path("assets/part1_results.csv")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
		if rows:
			writer.writeheader()
			writer.writerows(rows)
	print(f"[Part 1 Summary] Wrote {output_path}")
	return rows


def run_part2_all(profile: str, include_sophia: bool = False) -> None:
	configs = list(PART2_CONFIGS)
	if include_sophia:
		configs.append("configs/part2/sophia_bonus.yaml")
	for config_path in configs:
		print(f"\n=== Part 2 experiment: {config_path} ===")
		train_part2(config_path, profile=profile)


def train_tokenizer(config_path: str, save_dir: str) -> None:
	config = load_config(config_path)
	tokenizer = build_or_load_translation_tokenizer(
		config=config,
		tokenizer_dir=save_dir,
		num_train_samples=int(config.get("tokenizer_train_samples", 50000)),
	)
	print(f"Tokenizer ready: {len(tokenizer)} tokens at {save_dir}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Advanced NLP Assignment 2 experiments")
	subparsers = parser.add_subparsers(dest="command", required=True)

	train_tokenizer_parser = subparsers.add_parser("train-tokenizer")
	train_tokenizer_parser.add_argument("--config", required=True)
	train_tokenizer_parser.add_argument("--save-dir", default="./data/tokenizer_part1")

	train1_parser = subparsers.add_parser("train-part1", aliases=["part1-train"])
	train1_parser.add_argument("--config", required=True)
	train1_parser.add_argument("--profile", choices=("smoke", "full"), default=None)
	train1_parser.add_argument("--resume-from", default=None)

	eval1_parser = subparsers.add_parser("eval-part1", aliases=["part1-eval"])
	eval1_parser.add_argument("--config", required=True)
	eval1_parser.add_argument("--checkpoint", default=None)
	eval1_parser.add_argument("--profile", choices=("smoke", "full"), default=None)

	all1_parser = subparsers.add_parser("run-part1-all")
	all1_parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")

	train2_parser = subparsers.add_parser("train-part2", aliases=["part2-train"])
	train2_parser.add_argument("--config", required=True)
	train2_parser.add_argument("--profile", choices=("smoke", "full"), default=None)
	train2_parser.add_argument("--resume-from", default=None)
	train2_parser.add_argument("--sophia", action="store_true", help="Use the optional Sophia-H optimizer")

	eval2_parser = subparsers.add_parser("eval-part2", aliases=["part2-eval"])
	eval2_parser.add_argument("--config-dir", default="configs/part2")
	eval2_parser.add_argument("--output-plot", default="assets/part2_optimizer_comparison_v2.png")
	eval2_parser.add_argument("--output-json", default="assets/part2_optimizer_results_v2.json")
	eval2_parser.add_argument("--output-csv", default="assets/part2_optimizer_results_v2.csv")
	eval2_parser.add_argument("--include-sophia", action="store_true")

	all2_parser = subparsers.add_parser("run-part2-all")
	all2_parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
	all2_parser.add_argument("--include-sophia", action="store_true")

	eval3_parser = subparsers.add_parser("eval-part3", aliases=["part3-eval"])
	eval3_parser.add_argument("--config", required=True)
	eval3_parser.add_argument("--profile", choices=("smoke", "full"), default=None)

	return parser


def main() -> None:
	args = build_parser().parse_args()
	if args.command == "train-tokenizer":
		train_tokenizer(args.config, args.save_dir)
	elif args.command in {"train-part1", "part1-train"}:
		train_part1(args.config, profile=args.profile, resume_from=args.resume_from)
	elif args.command in {"eval-part1", "part1-eval"}:
		evaluate_part1(args.config, checkpoint_path=args.checkpoint, profile=args.profile)
	elif args.command == "run-part1-all":
		run_part1_all(args.profile)
	elif args.command in {"train-part2", "part2-train"}:
		train_part2(args.config, profile=args.profile, sophia=args.sophia, resume_from=args.resume_from)
	elif args.command == "run-part2-all":
		run_part2_all(args.profile, include_sophia=args.include_sophia)
	elif args.command in {"eval-part2", "part2-eval"}:
		evaluate_all_optimizers(
			args.config_dir, args.output_plot, args.output_json, args.output_csv,
			include_sophia=args.include_sophia,
		)
	elif args.command in {"eval-part3", "part3-eval"}:
		evaluate_part3(args.config, profile=args.profile)


if __name__ == "__main__":
	main()
