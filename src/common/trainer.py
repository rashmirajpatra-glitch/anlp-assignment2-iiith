import math
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn


def warmup_cosine_lr(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float = 0.1,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def manual_grad_norm_clip(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    norm = torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in gradients))
    norm_value = float(norm.item())
    if max_norm > 0 and norm_value > max_norm:
        scale = max_norm / (norm_value + 1e-6)
        for gradient in gradients:
            gradient.mul_(scale)
    return norm_value


class GenericTrainer:
    """Optimizer-agnostic single-step trainer with optional CUDA autocast."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        device: torch.device,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        before_step: Optional[Callable[[], None]] = None,
        before_backward: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device.type == "cuda"
        self.before_step = before_step
        self.before_backward = before_backward
        if before_backward is not None:
            # Sophia-H needs second derivatives. Flash and memory-efficient
            # SDPA kernels do not provide them, so use the math backend.
            for module in self.model.modules():
                if hasattr(module, "force_math_sdpa"):
                    module.force_math_sdpa = True
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

    def train_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype if self.use_amp else torch.float32,
            enabled=self.use_amp,
        ):
            model_kwargs = {"labels": labels}
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            output = self.model(input_ids, **model_kwargs)
            loss = output["loss"]
        if self.before_backward is not None:
            self.before_backward(loss)
        self.scaler.scale(loss).backward()
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = manual_grad_norm_clip(self.model.parameters(), self.grad_clip)
        if self.before_step is not None:
            self.before_step()
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "lm_loss": float(output["lm_loss"].detach().item()),
            "aux_loss": float(output["aux_loss"].detach().item()),
            "router_z_loss": float(output["router_z_loss"].detach().item()),
            "grad_norm": grad_norm,
        }
