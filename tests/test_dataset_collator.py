"""Tests for dataset and collator behavior."""

import json
import random
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from qwen3_rerank_trainer.training import (
    RerankDataset,
    RerankCollator,
    RLRerankDataset,
    RLCollator,
    StreamingRerankDataset,
    StreamingRLRerankDataset,
)
from qwen3_rerank_trainer.data.sampling import sample_documents, sample_documents_by_score


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"
        self.seen_padding_sides = []

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
                self.seen_padding_sides.append(self.padding_side)
                if self.padding_side == "left":
                    padded.append([self.pad_token_id] * pad_len + ids)
                    attention.append([0] * pad_len + [1] * len(ids))
                else:
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
    tokenizer = DummyTokenizer()
    collator = RerankCollator(
        tokenizer,
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
    assert tokenizer.padding_side == "right"
    assert "left" in tokenizer.seen_padding_sides


def test_rerank_collator_left_pads_and_restores_tokenizer():
    tokenizer = DummyTokenizer()
    collator = RerankCollator(
        tokenizer,
        max_length=10,
        format_fn=lambda q, d: d,
    )

    out = collator([
        {"query": "q1", "positives": ["a"], "negatives": ["b c"]},
    ])

    assert tokenizer.padding_side == "right"
    assert tokenizer.seen_padding_sides == ["left", "left"]
    assert out["input_ids"].tolist() == [[0, 1], [1, 2]]
    assert out["attention_mask"].tolist() == [[0, 1], [1, 1]]


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

    tokenizer = DummyTokenizer()
    collator = RLCollator(
        tokenizer,
        max_length=10,
        format_fn=lambda q, d: d,
    )
    out = collator([item])
    assert out["input_ids"].shape[0] == 3
    assert out["input_ids"].shape[1] == 1  # 最长 "p1"/"n1"/"n2" => 1 token
    assert out["labels"].shape[0] == 3
    assert out["group_sizes"] == [3]
    assert tokenizer.padding_side == "right"
    assert "left" in tokenizer.seen_padding_sides


def test_dataset_rejects_fixed_positive_count_not_smaller_than_group_size(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": "q1", "positives": ["p1", "p2", "p3"], "negatives": ["n1", "n2"]},
        ],
    )

    with pytest.raises(ValueError, match="n_pos must be smaller than n_docs"):
        RerankDataset(str(data_file), n_docs=2, n_pos=3)

    with pytest.raises(ValueError, match="n_pos must be smaller than n_docs"):
        RLRerankDataset(str(data_file), n_docs=2, n_pos=3)

    with pytest.raises(ValueError, match="n_pos must be 0 when n_docs=0"):
        StreamingRerankDataset(str(data_file), n_docs=0, n_pos=1)

    with pytest.raises(ValueError, match="n_pos must be 0 when n_docs=0"):
        StreamingRLRerankDataset(str(data_file), n_docs=0, n_pos=1)


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


def test_old_format_stringified_fields_are_normalized(tmp_path: Path):
    data_file = tmp_path / "old_format.jsonl"
    _write_jsonl(
        data_file,
        [
            {
                "query": "q1",
                "positives_strong": "['positive strong' 'positive medium']",
                "statement_very_hard_negatives": json.dumps(
                    [{"statement": "negative hard"}], ensure_ascii=False
                ),
                "statement_medium_negatives": "['negative medium' 'negative easy']",
            },
        ],
    )

    ds = RerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=0,
        max_length=16,
        seed=123,
    )
    assert len(ds) == 1
    assert ds.data[0]["positives"] == ["positive strong", "positive medium"]
    assert ds.data[0]["negatives"] == [
        "negative hard",
        "negative medium",
        "negative easy",
    ]

    item = ds[0]
    assert "positive strong" in item["positives"]
    assert "negative hard" in item["negatives"]

    rl_ds = RLRerankDataset(
        str(data_file),
        tokenizer=DummyTokenizer(),
        n_docs=0,
        max_length=16,
        seed=123,
    )
    assert rl_ds.data[0]["positives"] == ["positive strong", "positive medium"]
    assert rl_ds.data[0]["negatives"][0] == "negative hard"


def test_iter_data_for_worker_shards_stream(monkeypatch, tmp_path: Path):
    import qwen3_rerank_trainer.training.dataset as dataset_mod

    data_file = tmp_path / "train.jsonl"
    _write_jsonl(
        data_file,
        [
            {"query": f"q{i}", "positives": [f"p{i}"], "negatives": [f"n{i}"]}
            for i in range(5)
        ],
    )
    monkeypatch.setattr(
        dataset_mod,
        "get_worker_info",
        lambda: SimpleNamespace(id=1, num_workers=2),
    )

    rows = list(dataset_mod._iter_data_for_worker(str(data_file)))
    assert [row["query"] for row in rows] == ["q1", "q3"]


def test_seeded_sampling_does_not_mutate_global_random_state():
    docs = [
        {"text": "p1", "label": 1},
        {"text": "n1", "label": 0},
        {"text": "n2", "label": 0},
    ]

    random.seed(777)
    state_before = random.getstate()
    sample_documents(docs, n_total=2, n_pos=1, seed=123)
    assert random.getstate() == state_before


def test_score_sampling_backfills_underfilled_hard_easy_split():
    docs = [
        {"text": "p1", "label": 1, "score": 1.0},
        {"text": "n1", "label": 0, "score": 0.9},
        {"text": "n2", "label": 0, "score": 0.5},
        {"text": "n3", "label": 0, "score": 0.1},
    ]

    sampled_docs, sampled_labels = sample_documents_by_score(
        docs,
        n_total=4,
        n_pos=1,
        hard_ratio=1.0,
        shuffle=False,
        seed=123,
    )

    assert len(sampled_docs) == 4
    assert sampled_labels.count(1) == 1
    assert sampled_labels.count(0) == 3
