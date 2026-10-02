import json
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    metadata: Dict[str, Any],
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".safetensors":
        checkpoint_path = checkpoint_path.with_suffix(".safetensors")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    save_file(state_dict, str(checkpoint_path))
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    return checkpoint_path


def load_model_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".safetensors":
        checkpoint_path = checkpoint_path.with_suffix(".safetensors")
    state_dict = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(state_dict)
    metadata_path = checkpoint_path.with_suffix(".json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}