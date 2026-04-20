"""Tests for dataset and collator behavior."""

import json
from pathlib import Path

import torch

from qwen3_rerank_trainer.training import (
    RerankDataset,
    RerankCollator,
    RLRerankDataset,
    RLCollator,
    StreamingRerankDataset,
    StreamingRLRerankDataset,
)


class DummyTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts,
        max_length=None,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,  # noqa: ARG002 - keep HF-compatible signature
        pad_to_multiple_of=None,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        for text in texts:
            tokens = text.strip().split()
            ids = list(range(1, len(tokens) + 1))
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)

        max_len = max((len(ids) for ids in input_ids), default=0)
        if padding:
            if pad_to_multiple_of:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            padded = []
            attention = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [self.pad_token_id] * pad_len)
                attention.append([1] * len(ids) + [0] * pad_len)
            input_ids = padded
            attention_mask = attention
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_rerank_dataset_sampling(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "q1", "positives": ["p1", "p2"], "negatives": ["n1", "n2", "n3"]},
        ],
    )
    ds = RerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=3,
        n_pos=1,
        max_length=16,
        seed=123,
    )
    item = ds[0]
    assert len(item["positives"]) == 1
    assert len(item["negatives"]) == 2


def test_rerank_dataset_filter_overlength(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "q1", "positives": ["short"], "negatives": ["too long text here", "short neg"]},
        ],
    )
    ds = RerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=2,
        n_pos=1,
        max_length=2,
        seed=123,
        filter_overlength=True,
        format_fn=lambda q, d: d,
    )
    item = ds[0]
    assert "too long text here" not in item["negatives"]


def test_rerank_collator_dynamic_padding():
    collator = RerankCollator(
        DummyTokenizer(),
        max_length=10,
        format_fn=lambda q, d: d,
    )
    batch = [
        {"query": "q1", "positives": ["a b"], "negatives": ["c d e f"]},
        {"query": "q2", "positives": ["x y z"], "negatives": []},
    ]
    out = collator(batch)
    assert out["input_ids"].shape[0] == 3
    # 动态 padding：最大长度应为 4（"c d e f"）
    assert out["input_ids"].shape[1] == 4
    assert out["labels"].shape[0] == 3


def test_rl_dataset_and_collator(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "q1", "positives": ["p1"], "negatives": ["n1", "n2"]},
        ],
    )
    ds = RLRerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=3,
        n_pos=1,
        max_length=16,
        seed=123,
    )
    item = ds[0]
    assert len(item["documents"]) == 3
    assert len(item["labels"]) == 3

    collator = RLCollator(
        DummyTokenizer(),
        max_length=10,
        format_fn=lambda q, d: d,
    )
    out = collator([item])
    assert out["input_ids"].shape[0] == 3
    assert out["input_ids"].shape[1] == 1  # 最长 "p1"/"n1"/"n2" => 1 token
    assert out["labels"].shape[0] == 3
    assert out["group_sizes"] == [3]


def test_streaming_datasets(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "q1", "positives": ["p1"], "negatives": ["n1"]},
            {"query": "q2", "positives": ["p2"], "negatives": ["n2"]},
            {"query": "q3", "positives": ["p3"], "negatives": ["n3"]},
        ],
    )
    ds = StreamingRerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=2,
        n_pos=1,
        max_length=16,
        max_samples=2,
        seed=42,
    )
    items = list(ds)
    assert len(items) == 2

    rl_ds = StreamingRLRerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=2,
        n_pos=1,
        max_length=16,
        max_samples=2,
        seed=42,
    )
    rl_items = list(rl_ds)
    assert len(rl_items) == 2
