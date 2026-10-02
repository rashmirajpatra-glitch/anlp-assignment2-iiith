import torch

from src.part1.evaluate import compute_and_plot_expert_heatmaps
from src.part1.model import MoE, Transformer, TransformerConfig
from src.part3.evaluate import compute_generation_metric
from src.part3.decoding import (
    beam_search_decode,
    greedy_decode,
    top_k_filter_logits,
    top_p_filter_logits,
)
from src.utils import train_multilingual_tokenizer


def _config(variant: str, top_k: int) -> TransformerConfig:
    return TransformerConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        vocab_size=128,
        max_seq_len=64,
        dropout=0.0,
        moe_variant=variant,
        top_k=top_k,
    )


def test_part2_wandb_config_aliases_enable_remote_run(monkeypatch):
    import wandb
    from src.utils import init_wandb

    observed = {}

    def fake_init(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(wandb, "init", fake_init)
    result = init_wandb({"project": "optimizer-project", "entity": "student"})

    assert result is not None
    assert observed["project"] == "optimizer-project"
    assert observed["entity"] == "student"


def test_moe_parameter_budgets_within_one_percent():
    variants = {
        "dense": 1,
        "moe_4_1": 1,
        "moe_4_2": 2,
        "moe_shared": 1,
        "moe_4_2_active": 2,
    }
    stats = {
        name: Transformer(_config(name, top_k)).get_parameter_stats()
        for name, top_k in variants.items()
    }
    dense_total = stats["dense"]["total_params"]
    dense_ffn = stats["dense"]["total_ffn_per_layer"]
    for name in ("dense", "moe_4_1", "moe_4_2", "moe_shared", "moe_4_2_active"):
        print(name, stats[name])
    for name in ("moe_4_1", "moe_4_2", "moe_shared"):
        assert abs(stats[name]["total_params"] / dense_total - 1.0) <= 0.01
    expected_active_ratios = {
        "moe_4_1": 0.25,
        "moe_4_2": 0.50,
        "moe_shared": 0.50,
        "moe_4_2_active": 1.0,
    }
    for name, expected_ratio in expected_active_ratios.items():
        actual_ratio = stats[name]["active_expert_ffn_per_layer"] / dense_ffn
        assert abs(actual_ratio / expected_ratio - 1.0) <= 0.01
    assert abs(stats["moe_4_2_active"]["active_params"] / stats["dense"]["active_params"] - 1.0) <= 0.01


def test_moe_padding_positions_are_excluded_from_routing():
    moe = MoE(_config("moe_4_2", top_k=2)).eval()
    values = torch.randn(2, 3, 64)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    padded_output, aux_loss, _ = moe(values, attention_mask=mask)
    observed_indices = moe.last_routing_indices.clone()
    observed_probs = moe.last_router_probs.clone()

    compact_values = values[mask.bool()].unsqueeze(0)
    compact_output, compact_aux_loss, _ = moe(compact_values)

    assert observed_indices.shape[0] == int(mask.sum())
    assert observed_probs.shape[0] == int(mask.sum())
    assert torch.allclose(padded_output[mask.bool()], compact_output[0])
    assert torch.allclose(padded_output[~mask.bool()], torch.zeros_like(padded_output[~mask.bool()]))
    assert torch.allclose(aux_loss, compact_aux_loss)


def test_part1_token_budget_trimming_never_keeps_targetless_rows():
    from src.part1.train import _limit_batch_tokens

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "labels": torch.tensor([[-100, -100, 3, -100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }
    too_small = _limit_batch_tokens(batch, remaining_tokens=2, pad_id=0)
    assert too_small["input_ids"].numel() == 0

    # A batch with no target continuation (for example, all source tokens)
    # must never reach cross entropy.
    targetless = {**batch, "labels": torch.full_like(batch["labels"], -100)}
    assert _limit_batch_tokens(targetless, remaining_tokens=100, pad_id=0)["input_ids"].numel() == 0

    just_enough = _limit_batch_tokens(batch, remaining_tokens=3, pad_id=0)
    assert int((just_enough["labels"][:, 1:] != -100).sum()) == 1
    assert int(just_enough["attention_mask"].sum()) == 3


def test_tiny_part1_training_step_has_finite_loss_and_updates_weights():
    from src.common.trainer import GenericTrainer
    from src.part2.optimizers import CustomAdamW

    model = Transformer(_config("dense", top_k=1))
    optimizer = CustomAdamW(model.parameters(), lr=1e-3)
    trainer = GenericTrainer(model, optimizer, torch.device("cpu"), grad_clip=1.0)
    inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    labels = torch.tensor([[-100, 2, 3, -100], [-100, 5, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    before = model.tok_embeddings.weight.detach().clone()

    metrics = trainer.train_batch(inputs, labels, attention_mask=mask)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, model.tok_embeddings.weight)


def test_router_shapes_top_k_and_auxiliary_loss():
    moe = MoE(_config("moe_4_2", top_k=2))
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0].fill_(1.0)
        moe.router.weight[1:].fill_(-1.0)
    output, aux_loss, z_loss = moe(torch.ones(2, 4, 64))
    assert output.shape == (2, 4, 64)
    assert moe.last_routing_indices is not None
    assert moe.last_routing_indices.shape == (8, 2)
    assert torch.all(moe.last_routing_indices[:, 0] != moe.last_routing_indices[:, 1])
    assert aux_loss.item() > 0
    assert z_loss.item() > 0


def test_multilingual_tokenizer_round_trip(tmp_path):
    texts = ["Tôi đang học xử lý ngôn ngữ tự nhiên.", "私は自然言語処理を勉強しています。"]
    tokenizer = train_multilingual_tokenizer(iter(texts), tmp_path, vocab_size=300)
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(token_ids, skip_special_tokens=True) == text


def test_beam_width_one_matches_greedy():
    model = Transformer(
        TransformerConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            vocab_size=32,
            max_seq_len=32,
            dropout=0.0,
        )
    ).eval()
    prompts = (torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]]))
    for prompt in prompts:
        greedy = greedy_decode(model, prompt, max_new_tokens=5)
        observed_cache_flags = []
        original_forward = model.forward

        def recording_forward(*args, **kwargs):
            observed_cache_flags.append(kwargs.get("use_cache", False))
            return original_forward(*args, **kwargs)

        model.forward = recording_forward
        beam = beam_search_decode(model, prompt, max_new_tokens=5, beam_width=1, length_penalty=2.0)
        model.forward = original_forward
        assert torch.equal(greedy, beam)
        assert observed_cache_flags and all(observed_cache_flags)


def test_batched_beam_search_with_kv_cache():
    model = Transformer(
        TransformerConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            vocab_size=32,
            max_seq_len=32,
            dropout=0.0,
        )
    ).eval()
    prompt = torch.tensor([[1, 2, 3]])
    output = beam_search_decode(
        model,
        prompt,
        max_new_tokens=4,
        beam_width=2,
        eos_token_id=None,
    )
    assert output.shape == (1, 7)


def test_top_p_filtered_distribution_is_normalized():
    logits = torch.tensor([[4.0, 2.0, 1.0, -2.0]])
    probabilities = torch.softmax(top_p_filter_logits(logits, 0.8), dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1))
    assert probabilities[0, 0] > 0
    assert probabilities[0, -1] == 0


def test_top_k_filter_keeps_exactly_k_even_with_ties():
    logits = torch.tensor([[3.0, 2.0, 2.0, 2.0, 1.0]])
    filtered = top_k_filter_logits(logits, 2)
    assert torch.isfinite(filtered).sum().item() == 2


def test_all_manual_decoders_run_with_tiny_model():
    from src.part3.decoding import top_k_decode, top_p_decode

    model = Transformer(TransformerConfig(
        d_model=16, n_layers=1, n_heads=4, d_ff=64, vocab_size=32,
        max_seq_len=32, dropout=0.0,
    )).eval()
    prompt = torch.tensor([[1, 2, 3]])
    decoders = [
        greedy_decode(model, prompt, max_new_tokens=3),
        top_k_decode(model, prompt, max_new_tokens=3, k=4),
        top_p_decode(model, prompt, max_new_tokens=3, p=0.8),
        beam_search_decode(model, prompt, max_new_tokens=3, beam_width=2),
    ]
    assert all(result.shape[0] == 1 and 3 <= result.shape[1] <= 6 for result in decoders)


def test_part2_pretraining_dataset_chunks_all_tokens_without_duplicate_targets():
    from src.part2.train import PretrainingTextDataset

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def encode(self, text, add_special_tokens=False):
            return [int(token) for token in text.split()]

    dataset = PretrainingTextDataset(["1 2 3 4 5 6 7"], TinyTokenizer(), max_seq_len=4)
    assert len(dataset) == 3
    assert dataset.num_target_tokens == 7
    assert sum(int((dataset[index]["labels"][1:] != -100).sum()) for index in range(len(dataset))) == 7


def test_part2_dense_full_architecture_matches_part1_dense():
    from src.utils import load_config

    part1 = load_config("configs/part1/dense.yaml")
    part2 = load_config("configs/part2/adamw.yaml")
    for key in ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len"):
        assert part1[key] == part2[key]
    assert part1["dropout"] == part2.get("dropout", 0.1)


def test_generation_jsonl_writer_records_machine_readable_rows(tmp_path):
    import json
    from src.part3.evaluate import write_generation_jsonl

    path = tmp_path / "raw" / "greedy.jsonl"
    record = {"sample_id": 4, "prompt": "Once", "generated_continuation": " upon", "decoding_method": "greedy"}
    write_generation_jsonl([record], path)
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_rocstories_reference_token_accuracy_has_explicit_token_denominator():
    from src.part3.evaluate import compute_reference_token_accuracy

    class TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return {"prompt": [1, 2], "reference": [3, 4, 5]}[text]

    model = Transformer(TransformerConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16,
        vocab_size=8, max_seq_len=8, dropout=0.0,
    )).eval()
    result = compute_reference_token_accuracy(
        model,
        TinyTokenizer(),
        [{"prompt": "prompt", "reference": "reference"}],
        torch.device("cpu"),
    )
    assert result["total_tokens"] == 3
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "teacher-forced next-token argmax" in result["definition"]


def test_generation_metric_is_configurable():
    result = compute_generation_metric(["same", "other"], ["same", "ref"], "exact_match")
    assert result == {"metric": "exact_match", "value": 0.5}


def test_translation_tokenizer_is_fitted_only_on_training_split(tmp_path, monkeypatch):
    from datasets import Dataset, DatasetDict
    import src.part1.data as data_module

    data = DatasetDict({
        "train": Dataset.from_list([{"vi": "train vi", "ja": "train ja", "en": "train en"}]),
        "validation": Dataset.from_list([{"vi": "validation vi", "ja": "validation ja", "en": "validation en"}]),
        "test": Dataset.from_list([{"vi": "test vi", "ja": "test ja", "en": "test en"}]),
    })
    observed = []
    monkeypatch.setattr(data_module, "load_dataset", lambda *_args, **_kwargs: data)

    def capture_fit(texts_iterator, **_kwargs):
        observed.extend(texts_iterator)
        return "fitted"

    monkeypatch.setattr(data_module, "train_multilingual_tokenizer", capture_fit)
    result = data_module.build_or_load_translation_tokenizer(
        config={"dataset_name": "test-corpus", "vocab_size": 300},
        tokenizer_dir=tmp_path / "tokenizer",
        num_train_samples=50,
    )
    assert result == "fitted"
    assert set(observed) == {"train vi", "train ja", "train en"}


def test_top2_router_heatmap_fractions_sum_to_one(tmp_path):
    texts = ["Tôi đang học.", "私は勉強しています。"]
    tokenizer = train_multilingual_tokenizer(iter(texts), tmp_path / "tokenizer", vocab_size=300)
    model = Transformer(
        TransformerConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            vocab_size=len(tokenizer),
            max_seq_len=64,
            dropout=0.0,
            moe_variant="moe_4_2",
            top_k=2,
        )
    ).eval()
    result = compute_and_plot_expert_heatmaps(
        model,
        tokenizer,
        [
            {"src_lang": "vi", "src_text": texts[0]},
            {"src_lang": "ja", "src_text": texts[1]},
        ],
        torch.device("cpu"),
        tmp_path / "routing.png",
        max_samples=2,
    )
    assert result is not None
    assert torch.allclose(
        torch.tensor(result["freq_vi"]).sum(dim=-1), torch.ones(2), atol=1e-6
    )
    assert torch.allclose(
        torch.tensor(result["freq_ja"]).sum(dim=-1), torch.ones(2), atol=1e-6
    )
