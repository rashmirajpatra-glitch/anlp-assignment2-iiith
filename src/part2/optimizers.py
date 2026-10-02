import math
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


def _global_grad_clip_scale(parameters, max_norm: float) -> float:
    """Return one global gradient scale shared by every parameter in an optimizer."""
    if max_norm <= 0:
        return 1.0
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 1.0
    norm = float(squared_norm.sqrt().item())
    return min(1.0, max_norm / (norm + 1e-6))


class CustomAdamW(Optimizer):
    """
    Category 1: Standard Adaptive Baseline (AdamW)
    Reference: Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization".
    Category from 'Fantastic Pretraining Optimizers and Where to Find Them', Table 1.

    Implemented completely from scratch using torch.optim.Optimizer as the base class.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 1. Decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 2. Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 4. Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Mars(Optimizer):
    """Mars variance-reduced AdamW variant from Algorithm 5 of the assignment paper."""
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.98, 0.99),
        gamma: float = 0.05,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if gamma < 0.0 or grad_norm <= 0.0:
            raise ValueError("gamma must be nonnegative and grad_norm must be positive")

        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            grad_norm=grad_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        corrected_norm_sq = None
        for group in self.param_groups:
            beta1 = group["betas"][0]
            gamma = group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["previous_grad"] = torch.zeros_like(p.data)
                corrected = p.grad.detach() + gamma * beta1 / (1.0 - beta1) * (
                    p.grad.detach() - state["previous_grad"]
                )
                part = corrected.float().square().sum()
                corrected_norm_sq = part if corrected_norm_sq is None else corrected_norm_sq + part
        if corrected_norm_sq is None:
            grad_scale = 1.0
        else:
            corrected_norm = float(corrected_norm_sq.sqrt().item())
            max_norm = self.param_groups[0]["grad_norm"]
            grad_scale = min(1.0, max_norm / (corrected_norm + 1e-6))

        for group in self.param_groups:
            lr, (beta1, beta2), gamma, eps, weight_decay, grad_norm = (
                group[key]
                for key in ("lr", "betas", "gamma", "eps", "weight_decay", "grad_norm")
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["previous_grad"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                previous_grad = state["previous_grad"]

                state["step"] += 1
                step = state["step"]

                corrected_grad = grad + gamma * beta1 / (1.0 - beta1) * (grad - previous_grad)
                corrected_grad = corrected_grad * grad_scale

                exp_avg.mul_(beta1).add_(corrected_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(corrected_grad, corrected_grad, value=1.0 - beta2)

                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                direction = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                p.data.add_(direction, alpha=-lr)

                previous_grad.copy_(grad)

        return loss


VarianceReducedAdamW = Mars


class MemoryEfficientOptimizer(Optimizer):
    """
    Category 3: Memory-Efficient Optimizer (Lion: EvoLved Sign Momentum)
    Reference: Chen et al. (2023), "Symbolic Discovery of Optimization Algorithms" (Google Brain).
    Matches the Memory-Efficient category from 'Fantastic Pretraining Optimizers and Where to Find Them', Table 1.

    Lion tracks only ONE momentum buffer (50% reduction in optimizer state memory
    compared to AdamW), and computes parameter updates using element-wise sign operations.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
        grad_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, grad_norm=grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        parameters = [p for group in self.param_groups for p in group["params"]]
        grad_scale = _global_grad_clip_scale(parameters, self.param_groups[0]["grad_norm"])

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data * grad_scale

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)
                p.data.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz matrix iterations to compute the polar decomposition / orthogonalization
    of matrix G: returns an orthogonal matrix O such that O @ O^T = I.
    Pure native PyTorch matrix multiplications; no external libraries required.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    norm = torch.linalg.matrix_norm(X, ord="fro") + eps
    X = X / norm

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype=G.dtype)


class MatrixBasedOptimizer(Optimizer):
    """
    Category 4: Matrix-Based / Orthogonal Pretraining Optimizer (Muon)
    Reference: Jordan et al. (2024), "Muon: An optimizer for hidden layers in neural networks".
    Matches the Matrix-Based category from 'Fantastic Pretraining Optimizers and Where to Find Them', Table 1.

    For 2D weight matrices (linear projections in self-attention and FFN), Muon replaces
    standard coordinate-wise scaling with matrix orthogonalization via Newton-Schulz iterations.
    For 1D parameters (biases, layer norm scales), it falls back to standard momentum updates.
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        grad_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            grad_norm=grad_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        parameters = [p for group in self.param_groups for p in group["params"]]
        grad_scale = _global_grad_clip_scale(parameters, self.param_groups[0]["grad_norm"])

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data * grad_scale

                state = self.state[p]
                use_matrix_update = group.get("use_matrix", p.ndim >= 2)
                if use_matrix_update and p.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    nesterov_grad = momentum * buf + grad
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)
                    ortho_update = _newton_schulz_orthogonalize(nesterov_grad, steps=ns_steps)
                    scale = max(1.0, (p.size(0) / p.size(1)) ** 0.5)
                    p.data.add_(ortho_update, alpha=-lr * scale)
                else:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    correction1 = 1.0 - beta1 ** step
                    correction2 = 1.0 - beta2 ** step
                    denominator = (exp_avg_sq.sqrt() / math.sqrt(correction2)).add_(eps)
                    p.data.addcdiv_(exp_avg, denominator, value=-lr / correction1)

        return loss


class SophiaH(Optimizer):
    """
    Optional bonus optimizer: Sophia-H with a Hutchinson diagonal Hessian estimate.
    Reference: Liu et al. (2023), "Sophia: A Simple, Scalable, Robust Fast Optimizer
    for Pre-training Large Language Models".
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.96, 0.99),
        gamma: float = 0.04,
        hessian_update_interval: int = 10,
        eps: float = 1e-12,
        weight_decay: float = 0.1,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma <= 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if hessian_update_interval < 1:
            raise ValueError("hessian_update_interval must be positive")

        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            hessian_update_interval=hessian_update_interval,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def update_hessian(self, loss: torch.Tensor) -> None:
        """Estimate the Hessian diagonal with a Rademacher Hutchinson probe."""
        params = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        probes = [torch.empty_like(parameter).bernoulli_(0.5).mul_(2).sub_(1) for parameter in params]
        directional_derivative = sum(
            (gradient * probe).sum()
            for gradient, probe in zip(gradients, probes)
            if gradient is not None
        )
        hessian_products = torch.autograd.grad(
            directional_derivative,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        estimates = {
            parameter: (product * probe).detach().clamp_min(0.0)
            for parameter, product, probe in zip(params, hessian_products, probes)
            if product is not None
        }
        for group in self.param_groups:
            beta2 = group["betas"][1]
            interval = group["hessian_update_interval"]
            for p in group["params"]:
                if (self.state[p].get("step", 0)) % interval != 0:
                    continue
                estimate = estimates.get(p)
                if estimate is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p.data)
                h = state["hessian"]
                h.mul_(beta2).add_(estimate, alpha=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            gamma = group["gamma"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if "exp_avg" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                state["step"] += 1

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                denom = torch.clamp(gamma * hessian, min=eps)
                clipped_step = torch.clamp(exp_avg / denom, min=-1.0, max=1.0)

                p.data.add_(clipped_step, alpha=-lr)

        return loss
