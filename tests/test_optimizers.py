import math

import pytest
import torch

from src.part2.optimizers import (
    CustomAdamW,
    Mars,
    MatrixBasedOptimizer,
    MemoryEfficientOptimizer,
    SophiaH,
    VarianceReducedAdamW,
    _newton_schulz_orthogonalize,
)


def test_custom_adamw_matches_hand_computed_first_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = CustomAdamW(
        [parameter], lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1
    )
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    expected = 1.0 * (1.0 - 0.1 * 0.1) - 0.1 * 2.0 / (2.0 + 1e-8)
    assert math.isclose(parameter.item(), expected, rel_tol=0, abs_tol=1e-7)


def test_mars_first_step_uses_current_gradient_without_correction():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Mars(
        [parameter], lr=0.1, betas=(0.9, 0.99), gamma=0.5,
        weight_decay=0.0, grad_norm=100.0,
    )
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    assert math.isclose(parameter.item(), 0.9, rel_tol=0, abs_tol=1e-6)


def test_mars_clips_corrected_gradients_with_one_global_norm():
    first = torch.nn.Parameter(torch.tensor([0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = Mars(
        [first, second], lr=0.1, betas=(0.5, 0.5), gamma=0.0,
        weight_decay=0.0, grad_norm=1.0,
    )
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    optimizer.step()
    assert torch.allclose(optimizer.state[first]["exp_avg"], torch.tensor([0.3]))
    assert torch.allclose(optimizer.state[second]["exp_avg"], torch.tensor([0.4]))


def test_mars_applies_variance_correction_from_first_step():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = Mars(
        [parameter], lr=0.01, betas=(0.5, 0.5), gamma=1.0,
        weight_decay=0.0, grad_norm=100.0,
    )
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    assert torch.allclose(optimizer.state[parameter]["exp_avg"], torch.tensor([1.0]))


def test_matrix_optimizer_uses_adamw_fallback_group():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    embedding = torch.nn.Parameter(torch.ones(4, 2))
    optimizer = MatrixBasedOptimizer(
        [
            {"params": [matrix], "use_matrix": True},
            {"params": [embedding], "use_matrix": False},
        ],
        lr=0.01,
        weight_decay=0.0,
        ns_steps=2,
    )
    matrix.grad = torch.ones_like(matrix)
    embedding.grad = torch.ones_like(embedding)
    optimizer.step()
    assert "momentum_buffer" in optimizer.state[matrix]
    assert "exp_avg" not in optimizer.state[matrix]
    assert "exp_avg" in optimizer.state[embedding]
    assert "momentum_buffer" not in optimizer.state[embedding]


def test_muon_uses_nesterov_direction_before_orthogonalization():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = MatrixBasedOptimizer(
        [{"params": [parameter], "use_matrix": True}],
        lr=0.01, momentum=0.5, weight_decay=0.0, ns_steps=2, grad_norm=100.0,
    )
    parameter.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    optimizer.step()
    first_buffer = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    parameter.grad = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    before = parameter.detach().clone()
    optimizer.step()
    expected_momentum = 0.5 * first_buffer + parameter.grad
    expected_nesterov = 0.5 * expected_momentum + parameter.grad
    expected = before - 0.01 * _newton_schulz_orthogonalize(expected_nesterov, steps=2)
    assert torch.allclose(parameter, expected, atol=1e-6)


@pytest.mark.parametrize(
    "optimizer_type,kwargs",
    [
        (VarianceReducedAdamW, {"lr": 0.02}),
        (MemoryEfficientOptimizer, {"lr": 0.01, "weight_decay": 0.0}),
        (MatrixBasedOptimizer, {"lr": 0.01, "weight_decay": 0.0, "ns_steps": 2}),
        (SophiaH, {"lr": 0.01, "weight_decay": 0.0}),
    ],
)
def test_custom_optimizers_reduce_toy_quadratic(optimizer_type, kwargs):
    shape = (2, 2) if optimizer_type is MatrixBasedOptimizer else (2,)
    parameter = torch.nn.Parameter(torch.full(shape, 1.0))
    optimizer = optimizer_type([parameter], **kwargs)
    initial_loss = float(parameter.detach().square().mean())
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.square().mean() / 2
        if optimizer_type is SophiaH:
            optimizer.update_hessian(loss)
        loss.backward()
        optimizer.step()
    final_loss = float(parameter.detach().square().mean())
    assert final_loss < initial_loss


@pytest.mark.parametrize("optimizer_name", ["adamw", "mars", "memory_efficient", "matrix_based", "sophia"])
def test_each_part2_optimizer_trains_a_tiny_transformer_once(optimizer_name):
    from src.common.trainer import GenericTrainer
    from src.part1.model import Transformer, TransformerConfig
    from src.part2.train import get_optimizer_instance

    model = Transformer(TransformerConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16,
        vocab_size=20, max_seq_len=8, dropout=0.0,
    ))
    config = {
        "optimizer_name": optimizer_name,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "optimizer_grad_norm": 1.0,
        "ns_steps": 2,
        "hessian_update_interval": 1,
    }
    optimizer = get_optimizer_instance(optimizer_name, model, config)
    before_backward = optimizer.update_hessian if isinstance(optimizer, SophiaH) else None
    trainer = GenericTrainer(
        model, optimizer, torch.device("cpu"), grad_clip=1.0,
        before_backward=before_backward,
    )
    inputs = torch.tensor([[1, 2, 3, 4]])
    labels = inputs.clone()
    before = model.tok_embeddings.weight.detach().clone()

    metrics = trainer.train_batch(inputs, labels)

    assert math.isfinite(metrics["loss"])
    assert not torch.equal(before, model.tok_embeddings.weight)
