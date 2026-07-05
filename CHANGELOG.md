# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Corrected REINFORCE policy-gradient direction so negative advantages lower document
  yes scores instead of increasing them.
- Added validation for fixed positive sampling where `n_pos` must be smaller than `n_docs`.
- Preserved retryable API errors for async rerank retries and made sync API wrappers usable
  from notebooks or services with an existing event loop.
- Hardened MTEB rerank normalization against duplicate document indices and accepted
  `query`/`question` query fields in MTEB v2-style rows.
- Updated install hints and release tooling prerequisites.

## [0.2.1] - 2026-07-05

### Fixed

- **Training CLI**
  - Made SFT/RL command-line entrypoints fail cleanly on missing inputs before importing
    heavyweight optional trainer dependencies.
  - Added RL precision flags and removed the hard-coded bf16 training setting.
  - Added a `pyarrow<21` dependency guard for compatibility with supported `datasets`
    releases.
- **Data and Tokenization**
  - Preserved caller tokenizer padding state and enforced left-padding where trainer logits
    rely on the final token.
  - Fixed old-format stringified list loading and streaming dataset worker sharding.
  - Avoided global RNG mutation and improved negative-sampling backfill behavior.
- **Losses and RL**
  - Added shape/empty-input validation for RL rewards, REINFORCE, and DPO helpers.
  - Fixed tie-aware ListMLE and graph-connected no-valid-pair loss returns.
  - Corrected RLTrainer multi-iteration gradient scaling and reference-logit handling.
- **Evaluation and Reporting**
  - Normalized local/API reranker return contracts for MTEB evaluation.
  - Fixed MTEB v2 qrels-only fallback inflation, two-stage API rerank construction,
    NumPy JSON serialization, and zero-score report aggregation.
- **Docs and Tests**
  - Updated release workflow text, README examples, Chinese README, and regression coverage
    for CLI, data, loss, metric, and evaluation paths.

## [0.2.0] - 2026-06-05

### Added

- **Loss Wrappers**
  - Public `nn.Module` wrappers for LambdaLoss, ListMLE, ListNet/Listwise, RankNet,
    and pointwise yes/no cross-entropy losses.
  - Metric-specific LambdaLoss aliases for NDCG, MAP, and MRR.
- **RL Wrappers**
  - Public `nn.Module` wrappers for REINFORCE, GRPO, DAPO, Dr. GRPO, and DPO losses.
  - Public reward helper aliases for rank-based, score-based, NDCG-based, and
    recall-based rewards.
- **CI**
  - GitHub Actions workflow for test and build validation on Python 3.10 and 3.11.

### Changed

- **Release Workflow**
  - Clarified that `qwen3-rerank-trainer` is the source of truth and downstream
    repositories should consume released package versions instead of copying source.

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
