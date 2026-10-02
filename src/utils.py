import sys
import random
import json
import shutil
import tempfile
import yaml
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

# Ensure UTF-8 output streams on all operating systems (Windows PowerShell compatibility)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast


def get_device() -> torch.device:
    """
    Resolve the best available compute device.

    AMD GPUs (e.g. Radeon 8050S iGPU on Ryzen AI) have no ROCm support on
    Windows, so we use the `torch-directml` backend there instead. Falls back
    to CUDA (NVIDIA), then MPS (Apple), then CPU.
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, models, trainers, decoders


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration dictionary from a YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def init_wandb(
    config: Dict[str, Any],
    job_type: str = "train",
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """
    Initialize Weights & Biases tracking with safe placeholder handling.
    Empty project/entity settings disable remote tracking until the user configures them.
    """
    if str(config.get("profile", "full")).lower() != "full":
        return None

    import wandb

    # Part 1/3 configs use wandb_*; Part 2 configs use project/entity.
    # Accept both schemas so configured Part 2 runs are not silently disabled.
    project = config.get("wandb_project") or config.get("project", "TODO_FILL_ME")
    entity = config.get("wandb_entity") or config.get("entity", "TODO_FILL_ME")
    mode = config.get("wandb_mode", "online")

    is_placeholder = (
        not project
        or "TODO" in str(project)
        or not entity
        or "TODO" in str(entity)
    )

    if is_placeholder:
        print(
            "[WandB Notice] Project/entity are unset; external tracking is disabled for this run."
        )
        return None

    run = wandb.init(
        project=project if not is_placeholder else None,
        entity=entity if (not is_placeholder and entity != "TODO_FILL_ME") else None,
        name=run_name or config.get("run_name", None),
        job_type=job_type,
        tags=tags or config.get("tags", []),
        config=config,
        mode=mode,
        dir=config.get("wandb_dir", "./runs/wandb"),
    )
    return run


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Any],
    step: int,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    save_dir: Union[str, Path],
    filename: Optional[str] = None,
) -> Path:
    """Save model and optimizer state locally to a non-git-tracked directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_step_{step}.pt"
    ckpt_path = save_dir / filename
    raw_model = model.module if hasattr(model, "module") else model
    metadata = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "config": config,
    }
    from src.common.checkpoints import save_model_checkpoint

    safe_path = save_model_checkpoint(raw_model, ckpt_path, metadata)
    if optimizer is not None:
        optimizer_path = safe_path.with_name(f"{safe_path.stem}.optimizer.pt")
        torch.save({
            "optimizer_state_dict": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, optimizer_path)
    print(f"[Checkpoint Saved] Step {step} -> {safe_path}")
    return safe_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model and optimizer state from a local checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    safe_path = checkpoint_path.with_suffix(".safetensors")
    if safe_path.exists():
        checkpoint_path = safe_path
        from src.common.checkpoints import load_model_checkpoint

        state = load_model_checkpoint(checkpoint_path, model, device=device)
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(state["model_state_dict"])

    raw_model = model.module if hasattr(model, "module") else model

    optimizer_state = state.get("optimizer_state_dict")
    optimizer_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.optimizer.pt")
    auxiliary_state = None
    if optimizer is not None and optimizer_state is None and optimizer_path.exists():
        auxiliary_state = torch.load(optimizer_path, map_location=device, weights_only=False)
        if isinstance(auxiliary_state, dict) and "optimizer_state_dict" in auxiliary_state:
            optimizer_state = auxiliary_state["optimizer_state_dict"]
        else:  # Backward compatibility with earlier optimizer-only sidecars.
            optimizer_state = auxiliary_state
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    elif optimizer is not None:
        raise ValueError(
            f"Cannot resume training from {checkpoint_path}: optimizer state is missing. "
            "Choose a local training checkpoint with its .optimizer.pt sidecar."
        )

    if isinstance(auxiliary_state, dict) and optimizer is not None:
        if auxiliary_state.get("python_rng_state") is not None:
            random.setstate(auxiliary_state["python_rng_state"])
        if auxiliary_state.get("numpy_rng_state") is not None:
            np.random.set_state(auxiliary_state["numpy_rng_state"])
        if auxiliary_state.get("torch_rng_state") is not None:
            torch.set_rng_state(auxiliary_state["torch_rng_state"].cpu())
        cuda_rng_state = auxiliary_state.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

    print(f"[Checkpoint Loaded] Successfully restored from {checkpoint_path} (step {state.get('step', 'unknown')})")
    return state


def load_checkpoint_metadata(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return {key: value for key, value in state.items() if key != "model_state_dict"}
    raise FileNotFoundError(f"Checkpoint metadata not found for {checkpoint_path}")


def push_to_hf_hub(
    local_dir: Union[str, Path],
    repo_id: str,
    commit_message: str = "Upload model checkpoint",
    private: bool = False,
    token: Optional[str] = None,
) -> bool:
    """
    Safely push local model folder to Hugging Face Hub.
    Skips upload if repo_id is still a placeholder ('TODO_FILL_ME').
    """
    if "TODO" in repo_id or not repo_id.strip():
        print(f"[HuggingFace Hub] Skipped upload: repo_id contains placeholder ('{repo_id}').")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        uploaded_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        expected_files = {
            path.relative_to(Path(local_dir)).as_posix()
            for path in Path(local_dir).rglob("*")
            if path.is_file()
        }
        missing_files = expected_files - uploaded_files
        if missing_files:
            raise RuntimeError(f"Upload verification failed; missing files: {sorted(missing_files)}")
        print(f"[HuggingFace Hub] Successfully uploaded {local_dir} -> https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"[HuggingFace Hub Error] Failed to upload to {repo_id}: {e}")
        return False


def push_checkpoint_artifact_to_hub(
    checkpoint_path: Union[str, Path],
    repo_id: str,
    metadata: Dict[str, Any],
    tokenizer_dir: Optional[Union[str, Path]] = None,
    commit_message: str = "Upload final assignment checkpoint",
) -> bool:
    """Upload only a verified final model, its metadata, and optional tokenizer."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata does not exist: {metadata_path}")
    if tokenizer_dir is not None and not (Path(tokenizer_dir) / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Tokenizer is missing tokenizer.json: {tokenizer_dir}")

    with tempfile.TemporaryDirectory(prefix="anlp2-hf-upload-") as temporary_dir:
        staging_dir = Path(temporary_dir)
        shutil.copy2(checkpoint_path, staging_dir / "model.safetensors")
        shutil.copy2(metadata_path, staging_dir / "training_metadata.json")
        (staging_dir / "model_config.json").write_text(
            json.dumps(metadata.get("config", metadata), indent=2, default=str),
            encoding="utf-8",
        )
        (staging_dir / "README.md").write_text(
            "# Assignment 2 checkpoint\n\n"
            "This repository contains the final checkpoint produced by this project's "+
            "training pipeline. See `training_metadata.json` for the recorded model "+
            "configuration, profile, seed, and token count.\n",
            encoding="utf-8",
        )
        if tokenizer_dir is not None:
            shutil.copytree(Path(tokenizer_dir), staging_dir / "tokenizer")
        return push_to_hf_hub(
            staging_dir,
            repo_id,
            commit_message=commit_message,
            private=False,
        )


def build_bpe_tokenizer(
    vocab_size: int = 32000,
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """
    Build a Byte-Level BPE Tokenizer capable of handling mixed Vietnamese,
    Japanese (Hiragana, Katakana, Kanji), and English without romanization.
    Using ByteLevel pre-tokenization prevents out-of-vocabulary UNK tokens
    for arbitrary UTF-8 characters.
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<vi>", "<ja>", "<en>"]

    # BPE model with byte-level fallback
    model = models.BPE(unk_token="<unk>")
    tokenizer = Tokenizer(model)

    # Normalize with NFKC (standard for Japanese/multilingual Unicode consistency)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])

    # Isolate Japanese CJK characters before byte-level splitting so text without spaces
    # does not become one long pre-token.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(r"[\p{Han}\p{Hiragana}\p{Katakana}]"),
            behavior="isolated",
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])

    # Byte-level decoder ensures decoded bytes accurately reconstruct UTF-8 strings
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_multilingual_tokenizer(
    texts_iterator: Iterator[str],
    save_dir: Union[str, Path],
    vocab_size: int = 32000,
    special_tokens: Optional[List[str]] = None,
) -> PreTrainedTokenizerFast:
    """
    Train a multilingual Byte-Level BPE tokenizer on text iterator and save it.
    Wraps the trained tokenizer in HuggingFace PreTrainedTokenizerFast.
    """
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<vi>", "<ja>", "<en>"]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_tokenizer = build_bpe_tokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"[Tokenizer] Training Byte-Level BPE tokenizer (vocab_size={vocab_size})...")
    raw_tokenizer.train_from_iterator(texts_iterator, trainer=trainer)

    tokenizer_file = save_dir / "tokenizer.json"
    raw_tokenizer.save(str(tokenizer_file))

    # Wrap in HF PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast_tokenizer.add_special_tokens({
        "additional_special_tokens": [t for t in special_tokens if t not in ["<pad>", "<unk>", "<bos>", "<eos>"]]
    })
    fast_tokenizer.save_pretrained(str(save_dir))
    print(f"[Tokenizer] Saved trained tokenizer to {save_dir}")
    return fast_tokenizer


def load_multilingual_tokenizer(load_dir: Union[str, Path]) -> PreTrainedTokenizerFast:
    """Load a previously trained multilingual tokenizer."""
    load_dir = Path(load_dir)
    if not (load_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"No tokenizer.json found at {load_dir}. Please train the tokenizer first.")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(load_dir),
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    return tokenizer
