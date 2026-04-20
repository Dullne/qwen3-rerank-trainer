# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-01-31

### Added

- **RL Training**
  - DPO loss support in RL trainer/CLI with `beta` and `reference_free` options
  - `num_iterations` for multi-update per batch and `chunk_size` for memory-friendly forward
- **Datasets**
  - Streaming datasets for SFT/RL and streaming iter loader for large-scale data
  - Optional overlength filtering and deterministic worker seeding
- **Evaluation**
  - API client retries/backoff and async batch worker for large inference batches

### Changed

- **Collators**
  - Dynamic padding and optional `pad_to_multiple_of` for better GPU utilization
- **Losses**
  - InfoNCE multi-positive modes unified under a single interface

### Fixed

- **RL**
  - Reference model handling and metrics logging stability improvements

## [0.1.3] - 2025-01-30

### Added

- **Multi-format Data Loading**
  - Support for JSONL, JSON, Parquet, Arrow formats via HuggingFace datasets
  - CSV support with automatic list field parsing (JSON, Python, NumPy/HF styles)
  - Auto column normalization: `positive` → `positives`, `negative` → `negatives`
  - New `load_data()` function replaces `load_jsonl()`

### Changed

- **Breaking**: `load_jsonl` renamed to `load_data` (supports all formats now)

## [0.1.2] - 2025-01-29

### Added

- **API Client**
  - `inference_framework` parameter for framework-specific request handling
  - vLLM Qwen3-Reranker client-side prompt preformatting (`_format_vllm_request`)
  - Custom `instruction` parameter for vLLM requests

### Fixed

- **API Client**
  - Added support for SGLang array response format `[{index, score, document}]`
  - Previously only supported `{"results": [...]}` format, causing all-zero scores with SGLang

## [0.1.1] - 2025-01-28

### Fixed

- **MTEB Evaluation**
  - Added support for MTEB v2 dataset format (queries/corpus/relevant_docs/top_ranked)
  - T2Reranking, MMarcoReranking and other datasets now work correctly

## [0.1.0] - 2025-01-22

### Added

- **Loss Functions**
  - LambdaLoss framework with NDCG, MAP, MRR weighting schemes
  - Listwise losses: ListMLE, p-ListMLE, ListNet
  - Pairwise losses: RankNet
  - Contrastive losses: InfoNCE, multi-positive InfoNCE
  - Pointwise losses: BCE, CE from yes/no logits

- **RL Training**
  - REINFORCE loss with doc-level rewards
  - DPO loss
  - Reward computation: rank-based, score-based, NDCG-based, recall-based

- **Evaluation**
  - Ranking metrics: MRR, AP, NDCG@k, P@k, R@k, Hit@k
  - MTEB reranking evaluation runner
  - Two-stage evaluation (Embedding + Rerank)
  - Report generation

- **Inference**
  - Qwen3Reranker class for model inference

- **Data Processing**
  - Input formatting for Qwen3-Reranker
  - Document sampling strategies
  - Tokenization utilities
