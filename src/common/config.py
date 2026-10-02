from typing import Any, Dict, Optional

from src.part1.model import TransformerConfig
from src.utils import load_config


def load_transformer_config(
    config_path: str,
    vocab_size: Optional[int] = None,
) -> TransformerConfig:
    values: Dict[str, Any] = load_config(config_path)
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    accepted = TransformerConfig.__dataclass_fields__
    model_values = {key: value for key, value in values.items() if key in accepted}
    return TransformerConfig(**model_values)