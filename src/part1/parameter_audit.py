"""Lightweight, allocation-free Part 1 parameter-matching audit."""

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from src.part1.model import Transformer, TransformerConfig
from src.utils import load_config


CONFIGS = {
    "dense": "dense.yaml",
    "top1": "moe_top1.yaml",
    "top2": "moe_top2.yaml",
    "shared_plus_routed": "moe_shared_routed.yaml",
    "top2_matched_active": "moe_top2_matched_active.yaml",
}
MATCH_TOLERANCE_PERCENT = 1.0


def _model_config(config: Dict[str, Any]) -> TransformerConfig:
    return TransformerConfig(
        d_model=int(config.get("d_model", 512)),
        n_layers=int(config.get("n_layers", 8)),
        n_heads=int(config.get("n_heads", 8)),
        d_ff=int(config.get("d_ff", 2048)),
        vocab_size=int(config.get("vocab_size", 16384)),
        max_seq_len=int(config.get("max_seq_len", 256)),
        dropout=float(config.get("dropout", 0.0)),
        use_moe=bool(config.get("use_moe", False)),
        moe_variant=str(config.get("moe_variant", "dense")),
        n_experts=int(config.get("n_experts", 4)),
        top_k=int(config.get("top_k", 1)),
    )


def audit_parameter_matching(config_dir: str = "configs/part1") -> Dict[str, Dict[str, int]]:
    """Instantiate models on the meta device and report actual shape-based counts."""
    results = {}
    for variant, filename in CONFIGS.items():
        config = load_config(Path(config_dir) / filename)
        with torch.device("meta"):
            model = Transformer(_model_config(config))
        stats = model.get_parameter_stats()
        stats["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results[variant] = {key: int(value) for key, value in stats.items()}

    dense = results["dense"]
    print("variant                  total parameters   trainable   active   total delta   active delta")
    for variant, stats in results.items():
        total_delta = 100 * (stats["total_params"] - dense["total_params"]) / max(1, dense["total_params"])
        active_delta = 100 * (stats["active_params"] - dense["active_params"]) / max(1, dense["active_params"])
        print(
            f"{variant:24} {stats['total_params']:17,d} {stats['trainable_params']:11,d} "
            f"{stats['active_params']:8,d} {total_delta:11.3f}% {active_delta:12.3f}%"
        )
        compared = stats["active_params"] if variant == "top2_matched_active" else stats["total_params"]
        reference = dense["active_params"] if variant == "top2_matched_active" else dense["total_params"]
        delta = abs(100 * (compared - reference) / max(1, reference))
        if variant != "dense" and delta > MATCH_TOLERANCE_PERCENT:
            metric = "active" if variant == "top2_matched_active" else "total"
            raise RuntimeError(
                f"{variant} {metric} parameter mismatch is {delta:.3f}%, "
                f"exceeding {MATCH_TOLERANCE_PERCENT:.1f}% tolerance"
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs/part1")
    args = parser.parse_args()
    audit_parameter_matching(args.config_dir)
