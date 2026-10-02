from pathlib import Path
import random
import hashlib
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, Dataset as HFDataset, DatasetDict

from src.utils import train_multilingual_tokenizer, load_multilingual_tokenizer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<vi>", "<ja>", "<en>"]


class TranslationPairDataset(Dataset):
    """
    PyTorch Dataset for translation pairs:
    Yields samples formatted as:
      Vi->En: '<vi> ' + vi_text + ' <en> ' + en_text + ' <eos>'
      Ja->En: '<ja> ' + ja_text + ' <en> ' + en_text + ' <eos>'
    Supports prompt masking (labels set to -100 for source prompt tokens).
    """
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        mask_prompt: bool = True,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt

        self.pad_token_id = tokenizer.pad_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id or 3
        self.vi_token_id = tokenizer.convert_tokens_to_ids("<vi>")
        self.ja_token_id = tokenizer.convert_tokens_to_ids("<ja>")
        self.en_token_id = tokenizer.convert_tokens_to_ids("<en>")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        src_lang = item["src_lang"]      # "vi" or "ja"
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]

        bos_id = self.tokenizer.bos_token_id or 2
        src_lang_id = self.vi_token_id if src_lang == "vi" else self.ja_token_id
        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False)

        # Reserve a target span and EOS, even when the source is long.
        max_seq_len = max(4, self.max_seq_len)
        fixed_prefix = [bos_id, src_lang_id]
        reserved_target = min(32, max_seq_len - len(fixed_prefix) - 2)
        source_capacity = max_seq_len - len(fixed_prefix) - 1 - reserved_target - 1
        src_ids = src_ids[:max(0, source_capacity)]
        prefix_ids = fixed_prefix + src_ids + [self.en_token_id]
        target_ids = tgt_ids[:max_seq_len - len(prefix_ids) - 1]
        full_ids = prefix_ids + target_ids + [self.eos_token_id]
        prefix = f"<bos> <{src_lang}> {src_text} <en>"

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        if self.mask_prompt:
            prefix_len = min(len(prefix_ids), len(labels))
            labels[:prefix_len] = -100  # Mask prompt tokens from loss calculation

        return {
            "input_ids": input_ids,
            "labels": labels,
            "src_lang": src_lang,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "prefix_text": prefix,
        }


def translation_collate_fn(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    """Collate function that dynamically pads sequences to the maximum length in the batch."""
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    src_langs = []
    src_texts = []
    tgt_texts = []
    prefix_texts = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        # Pad input_ids with pad_token_id
        padded_inputs = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        # Pad labels with -100 (ignored by cross entropy)
        padded_labels = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])

        batch_input_ids.append(padded_inputs)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(attention_mask)
        src_langs.append(item["src_lang"])
        src_texts.append(item["src_text"])
        tgt_texts.append(item["tgt_text"])
        prefix_texts.append(item["prefix_text"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
        "src_langs": src_langs,
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "prefix_texts": prefix_texts,
    }


def extract_translation_pairs(
    dataset_name: str = "belumind/en-vi-ja-curated-500k-triplets",
    split: str = "train",
    max_samples: Optional[int] = None,
    allow_synthetic: bool = False,
) -> List[Dict[str, str]]:
    """
    Load dataset and extract 2 translation pairs per triplet:
    1. Vietnamese -> English
    2. Japanese -> English
    Handles schema column variations robustly.
    """
    print(f"[Data] Loading dataset '{dataset_name}' (split='{split}')...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError(
                f"Could not load dataset '{dataset_name}' split '{split}'. "
                "Check the dataset ID, network access, and Hugging Face permissions."
            ) from e
        print(f"[Data Warning] Using synthetic smoke data because '{dataset_name}' is unavailable: {e}")
        ds = HFDataset.from_list(_synthetic_triplets(max_samples))

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    pairs: List[Dict[str, str]] = []
    col_names = ds.column_names
    vi_col = next((c for c in ["vi", "vietnamese", "src_vi"] if c in col_names), None)
    ja_col = next((c for c in ["ja", "japanese", "src_ja"] if c in col_names), None)
    en_col = next((c for c in ["en", "english", "tgt_en", "target"] if c in col_names), None)

    nested_col = next((c for c in ["translation", "translations"] if c in col_names), None)
    if not nested_col and not (vi_col and ja_col and en_col):
        if len(col_names) >= 3:
            vi_col, ja_col, en_col = col_names[0], col_names[1], col_names[2]
        else:
            raise ValueError(f"Unexpected dataset columns: {col_names}")

    for row in ds:
        nested = row.get(nested_col, {}) if nested_col else {}
        vi_value = row.get(vi_col) if vi_col else None
        ja_value = row.get(ja_col) if ja_col else None
        en_value = row.get(en_col) if en_col else None
        if isinstance(nested, dict):
            vi_value = vi_value or next((nested[k] for k in ("vi", "vietnamese", "src_vi") if k in nested), None)
            ja_value = ja_value or next((nested[k] for k in ("ja", "japanese", "src_ja") if k in nested), None)
            en_value = en_value or next((nested[k] for k in ("en", "english", "tgt_en") if k in nested), None)
        vi_text = str(vi_value or "").strip()
        ja_text = str(ja_value or "").strip()
        en_text = str(en_value or "").strip()

        if vi_text and en_text:
            pairs.append({"src_lang": "vi", "src_text": vi_text, "tgt_text": en_text})
        if ja_text and en_text:
            pairs.append({"src_lang": "ja", "src_text": ja_text, "tgt_text": en_text})

    print(f"[Data] Extracted {len(pairs)} total translation pairs ({len(pairs)//2} Vi-En, {len(pairs)//2} Ja-En).")
    return pairs


def _synthetic_triplets(max_samples: Optional[int]) -> List[Dict[str, str]]:
    examples = [
        {"vi": "Tôi đang học xử lý ngôn ngữ tự nhiên.", "ja": "私は自然言語処理を勉強しています。", "en": "I am studying natural language processing."},
        {"vi": "Mô hình ngôn ngữ lớn đang phát triển nhanh chóng.", "ja": "大規模言語モデルは急速に発展しています。", "en": "Large language models are developing rapidly."},
        {"vi": "Hôm nay thời tiết rất đẹp.", "ja": "今日はとても良い天気です。", "en": "The weather is very nice today."},
        {"vi": "Chúng tôi đánh giá một bộ tối ưu hóa mới.", "ja": "新しいオプティマイザを評価します。", "en": "We evaluate a new optimizer."},
    ]
    target_size = max(8, min(max_samples or 64, 256))
    return (examples * ((target_size + len(examples) - 1) // len(examples)))[:target_size]


def build_or_load_translation_tokenizer(
    config: Dict[str, Any],
    dataset_name: str = "belumind/en-vi-ja-curated-500k-triplets",
    tokenizer_dir: Union[str, Path] = "./data/tokenizer_part1",
    num_train_samples: int = 50000,
) -> PreTrainedTokenizerFast:
    """
    Ensure the Byte-Level BPE multilingual tokenizer is trained and saved.
    If already existing on disk, loads directly.
    """
    tokenizer_dir = Path(tokenizer_dir)
    dataset_name = str(config.get("dataset_name", dataset_name))
    vocab_size = int(config.get("vocab_size", 32000))
    # Fit only on the actual training partition. A sidecar fingerprint prevents
    # reuse of tokenizers created by older code before the split was applied.
    try:
        datasets = load_dataset(dataset_name)
        if isinstance(datasets, HFDataset):
            datasets = DatasetDict({"train": datasets})
        train_rows = datasets["train"]
        if datasets.get("validation", datasets.get("val")) is None or datasets.get("test") is None:
            indices = list(range(len(train_rows)))
            random.Random(int(config.get("seed", 42))).shuffle(indices)
            size = len(indices)
            val_count = max(1, int(round(size * float(config.get("val_ratio", 0.01))))) if size >= 3 else 0
            test_count = max(1, int(round(size * float(config.get("test_ratio", 0.01))))) if size >= 3 else 0
            train_count = max(1, size - val_count - test_count) if size else 0
            train_rows = train_rows.select(indices[:train_count])
        if config.get("max_samples") is not None:
            train_rows = train_rows.select(range(min(len(train_rows), int(config["max_samples"]))))
        train_pairs = _pairs_from_dataset(train_rows)
    except Exception as exc:
        if not config.get("allow_synthetic_data", False):
            raise RuntimeError(f"Could not load training split to fit tokenizer for '{dataset_name}'.") from exc
        train_pairs = _pairs_from_dataset(HFDataset.from_list(_synthetic_triplets(num_train_samples)))
    train_pairs = train_pairs[:num_train_samples]
    fingerprint = hashlib.sha256(
        "\n".join(p["src_text"] + "\0" + p["tgt_text"] for p in train_pairs).encode("utf-8")
    ).hexdigest()
    marker_path = tokenizer_dir / "training_fingerprint.json"
    if (tokenizer_dir / "tokenizer.json").exists() and marker_path.exists():
        cached_tokenizer = load_multilingual_tokenizer(tokenizer_dir)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if len(cached_tokenizer) == vocab_size and marker.get("sha256") == fingerprint:
            print(f"[Tokenizer] Loading cached tokenizer from {tokenizer_dir}...")
            return cached_tokenizer
        print(
            f"[Tokenizer] Cached vocabulary has {len(cached_tokenizer)} tokens, "
            f"but this profile requests {vocab_size}; retraining."
        )

    print(f"[Tokenizer] No existing tokenizer found at {tokenizer_dir}. Training from dataset...")
    def text_gen() -> Iterator[str]:
        for p in train_pairs:
            yield p["src_text"]
            yield p["tgt_text"]

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = train_multilingual_tokenizer(
        texts_iterator=text_gen(),
        save_dir=tokenizer_dir,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    marker_path.write_text(json.dumps({"sha256": fingerprint, "source": "train_split_only"}), encoding="utf-8")
    return tokenizer


def get_translation_dataloaders(
    config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create Train, Validation, and Test DataLoaders for Part 1.
    Splits data according to train_ratio, val_ratio, test_ratio.
    """
    dataset_name = config.get("dataset_name", "belumind/en-vi-ja-curated-500k-triplets")
    max_samples = config.get("max_samples", None)
    max_seq_len = config.get("max_seq_len", 512)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 0)

    allow_synthetic = bool(config.get("allow_synthetic_data", False))
    try:
        datasets = load_dataset(dataset_name)
        if isinstance(datasets, HFDataset):
            datasets = DatasetDict({"train": datasets})
    except Exception as exc:
        if not allow_synthetic:
            raise RuntimeError(
                f"Could not load dataset '{dataset_name}'. Check its Hub ID, network access, "
                "and Hugging Face permissions."
            ) from exc
        print(f"[Data Warning] Using synthetic smoke data because '{dataset_name}' is unavailable: {exc}")
        datasets = DatasetDict({"train": HFDataset.from_list(_synthetic_triplets(max_samples))})

    if "train" not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' has no train split; found {list(datasets)}")
    train_rows = datasets["train"]
    val_rows = datasets.get("validation", datasets.get("val"))
    test_rows = datasets.get("test")

    if val_rows is None or test_rows is None:
        indices = list(range(len(train_rows)))
        random.Random(int(config.get("seed", 42))).shuffle(indices)
        size = len(indices)
        val_count = max(1, int(round(size * float(config.get("val_ratio", 0.01))))) if size >= 3 else 0
        test_count = max(1, int(round(size * float(config.get("test_ratio", 0.01))))) if size >= 3 else 0
        train_count = max(1, size - val_count - test_count) if size else 0
        if val_rows is None:
            val_rows = train_rows.select(indices[train_count:train_count + val_count])
        if test_rows is None:
            test_rows = train_rows.select(indices[train_count + val_count:])
        train_rows = train_rows.select(indices[:train_count])

    if max_samples is not None:
        train_rows = train_rows.select(range(min(len(train_rows), max_samples)))
        val_rows = val_rows.select(range(min(len(val_rows), max_samples)))
        test_rows = test_rows.select(range(min(len(test_rows), max_samples)))

    train_pairs = _pairs_from_dataset(train_rows)
    val_pairs = _pairs_from_dataset(val_rows)
    test_pairs = _pairs_from_dataset(test_rows)
    random.Random(int(config.get("seed", 42))).shuffle(train_pairs)

    print(f"[Data Splits] Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    train_ds = TranslationPairDataset(train_pairs, tokenizer, max_seq_len=max_seq_len, mask_prompt=True)
    val_ds = TranslationPairDataset(val_pairs, tokenizer, max_seq_len=max_seq_len, mask_prompt=True)
    test_ds = TranslationPairDataset(test_pairs, tokenizer, max_seq_len=max_seq_len, mask_prompt=True)

    pad_id = tokenizer.pad_token_id or 0
    collate = lambda b: translation_collate_fn(b, pad_token_id=pad_id)

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def _pairs_from_dataset(dataset: HFDataset) -> List[Dict[str, str]]:
    columns = dataset.column_names
    vi_col = next((name for name in ("vi", "vietnamese", "src_vi") if name in columns), None)
    ja_col = next((name for name in ("ja", "japanese", "src_ja") if name in columns), None)
    en_col = next((name for name in ("en", "english", "tgt_en", "target") if name in columns), None)
    nested_col = next((name for name in ("translation", "translations") if name in columns), None)
    if not nested_col and not (vi_col and ja_col and en_col) and len(columns) >= 3:
        vi_col, ja_col, en_col = columns[:3]
    if not nested_col and not (vi_col and ja_col and en_col):
        raise ValueError(f"Unsupported translation dataset schema: {columns}")

    pairs: List[Dict[str, str]] = []
    for row in dataset:
        nested = row.get(nested_col, {}) if nested_col else {}
        vi = row.get(vi_col) if vi_col else None
        ja = row.get(ja_col) if ja_col else None
        en = row.get(en_col) if en_col else None
        if isinstance(nested, dict):
            vi = vi or next((nested[key] for key in ("vi", "vietnamese", "src_vi") if key in nested), None)
            ja = ja or next((nested[key] for key in ("ja", "japanese", "src_ja") if key in nested), None)
            en = en or next((nested[key] for key in ("en", "english", "tgt_en") if key in nested), None)
        vi_text, ja_text, en_text = str(vi or "").strip(), str(ja or "").strip(), str(en or "").strip()
        if vi_text and en_text:
            pairs.append({"src_lang": "vi", "src_text": vi_text, "tgt_text": en_text})
        if ja_text and en_text:
            pairs.append({"src_lang": "ja", "src_text": ja_text, "tgt_text": en_text})
    return pairs
