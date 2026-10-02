import torch

from src.common.checkpoints import load_model_checkpoint, save_model_checkpoint
from src.part1.model import Transformer, TransformerConfig
from src.part2.optimizers import CustomAdamW
from src.utils import load_checkpoint, save_checkpoint


def test_safetensors_checkpoint_round_trip(tmp_path):
    config = TransformerConfig(
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        vocab_size=32,
        max_seq_len=16,
        dropout=0,
    )
    model = Transformer(config)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint_path = save_model_checkpoint(
        model,
        tmp_path / "model.pt",
        {"step": 7, "config": {"d_model": 16}},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    metadata = load_model_checkpoint(checkpoint_path, model)
    assert checkpoint_path.suffix == ".safetensors"
    assert metadata["step"] == 7
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_training_checkpoint_restores_optimizer_and_rng(tmp_path):
    model = Transformer(TransformerConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16,
        vocab_size=16, max_seq_len=8, dropout=0.0,
    ))
    optimizer = CustomAdamW(model.parameters(), lr=1e-3)
    sample = next(model.parameters())
    sample.grad = torch.ones_like(sample)
    optimizer.step()
    saved = save_checkpoint(model, optimizer, 3, 0, 0.5, {"tokens_seen": 17}, tmp_path)
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_step = optimizer.state[sample]["step"]
    expected_random = torch.rand(1)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.manual_seed(1234)
    state = load_checkpoint(saved, model, optimizer)
    observed_random = torch.rand(1)

    assert state["step"] == 3
    assert optimizer.state[sample]["step"] == expected_step
    assert torch.equal(observed_random, expected_random)
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_model[name])
