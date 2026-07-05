# qwen3-rerank-trainer

<p align="center">
  <strong>English</strong> | <a href="README_zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/Dullne/qwen3-rerank-trainer/stargazers"><img src="https://img.shields.io/github/stars/Dullne/qwen3-rerank-trainer?style=social" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Dullne/qwen3-rerank-trainer" alt="License"></a>
</p>

Training and evaluation toolkit for Qwen3-Reranker.

## Installation

```bash
# Basic installation
pip install qwen3-rerank-trainer

# With inference support
pip install "qwen3-rerank-trainer[inference]"

# With MTEB evaluation
pip install "qwen3-rerank-trainer[eval]"

# With two-stage evaluation (Embedding + Rerank)
pip install "qwen3-rerank-trainer[evalscope]"

# Full installation
pip install "qwen3-rerank-trainer[full]"
```

For development from source:

```bash
git clone https://github.com/Dullne/qwen3-rerank-trainer.git
cd qwen3-rerank-trainer
pip install -e ".[full]"
```

## Features

### Loss Functions

```python
from qwen3_rerank_trainer import (
    lambda_loss,           # LambdaLoss framework
    list_mle,              # ListMLE
    infonce_loss,          # InfoNCE
    ranknet_loss,          # RankNet
)
from qwen3_rerank_trainer.losses import NDCGLoss2PPScheme

# LambdaLoss with NDCG optimization
loss = lambda_loss(scores, labels, metric="ndcg")
```

Note: `lambda_rank_loss` and `lambda_loss_ndcg/map/mrr` have been merged into `lambda_loss`.

### SFT Training

```bash
# Command line (requires pip install "qwen3-rerank-trainer[full]")
qwen3-rerank-train --model /path/to/Qwen3-Reranker-4B --data train.jsonl --output outputs/sft

# With LoRA
qwen3-rerank-train --model /path/to/model --data train.jsonl --output outputs/sft \
    --lora --lora-r 8 --lora-alpha 16 --n-docs 8 --n-pos 1

# Loss function options
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type infonce --temperature 0.05
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type infonce --infonce-mode posset
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type lambda_loss --lambda-metric ndcg
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type list_mle
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type ranknet
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type ranknet --ranknet-max-pairs 2000000

# Optional: filter over-length samples (disabled by default)
qwen3-rerank-train --model /path/to/model --data train.jsonl --filter-overlength
```

```python
# Python API
from qwen3_rerank_trainer import (
    RerankDataset,
    StreamingRerankDataset,
    RerankCollator,
    ContrastiveSFTTrainer,
)
from qwen3_rerank_trainer.training.sft_trainer import get_yes_no_token_ids

# Load dataset
dataset = RerankDataset(
    "train.jsonl",
    tokenizer=tokenizer,
    n_docs=8,
    n_pos=1,  # fixed 1 positive + 7 negatives
)

# Use the streaming dataset for large files to avoid high memory usage
# dataset = StreamingRerankDataset("train.jsonl", tokenizer=tokenizer, n_docs=8, n_pos=1)

# Create collator
collator = RerankCollator(tokenizer, max_length=4096)

# Get yes/no token IDs
yes_id, no_id = get_yes_no_token_ids(tokenizer)

# Create trainer
trainer = ContrastiveSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    yes_token_id=yes_id,
    no_token_id=no_id,
    chunk_size=16,  # chunked forward pass to save VRAM
)
trainer.train()
```

### RL Training

```bash
# Command line (requires pip install "qwen3-rerank-trainer[full]")
# Basic RL training (run SFT first)
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --output outputs/rl

# Use all documents (recommended for large-scale data)
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl \
    --n_docs 0 --max_docs 50  # use all docs, capped at 50 per sample

# Chunked forward pass (saves VRAM and supports arbitrary n_docs)
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl \
    --chunk_size 8  # process 8 docs per chunk

# DAPO loss (default)
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dapo

# Dr. GRPO loss
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dr_grpo

# DPO loss
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dpo --dpo_beta 0.1

# Optional: filter over-length samples (disabled by default)
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --filter-overlength
```

```python
# Python API
from qwen3_rerank_trainer.training import (
    RLRerankDataset,
    StreamingRLRerankDataset,
    RLCollator,
    RLTrainer,
    load_sft_model,
)

# Load the SFT model
model = load_sft_model("outputs/sft/final", "Qwen/Qwen3-Reranker-4B")

# Add a new LoRA adapter
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)

# Prepare the dataset
dataset = RLRerankDataset(
    "train.jsonl",
    tokenizer=tokenizer,
    n_docs=8,       # 8 documents per group
    n_pos=0,        # dynamically follow the original positive ratio
    max_docs=50,    # cap extreme samples
)

# Use the streaming dataset for large files
# dataset = StreamingRLRerankDataset("train.jsonl", tokenizer=tokenizer, n_docs=8, n_pos=0, max_docs=50)

# Create the collator
collator = RLCollator(tokenizer, max_length=4096)

# Create the trainer
trainer = RLTrainer(
    yes_token_id=tokenizer.convert_tokens_to_ids("yes"),
    no_token_id=tokenizer.convert_tokens_to_ids("no"),
    kl_coef=0.1,           # KL penalty coefficient
    reward_type="rank_based",  # rank_based, ndcg_based, recall_based
    loss_type="dapo",      # grpo, dapo, dr_grpo
    chunk_size=8,          # chunked forward pass to save VRAM
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train()
```

```python
# Low-level API
from qwen3_rerank_trainer import (
    reinforce_loss,
    dpo_loss,
    compute_doc_level_rewards,
    compute_doc_level_advantages,
)

# Compute rewards and advantages from per-document scores and labels
rewards = compute_doc_level_rewards(scores, labels, reward_type="ndcg_based")
advantages = compute_doc_level_advantages(scores, labels, reward_type="ndcg_based")

# REINFORCE loss from yes/no logits
loss, advantages, rewards, kl = reinforce_loss(yes_logits, no_logits, labels)
```

### Evaluation

```python
from qwen3_rerank_trainer import mrr, ndcg_at_k, compute_all_metrics

# Basic metrics
mrr_score = mrr(ranking, positive_indices)
ndcg_score = ndcg_at_k(ranking, relevance_scores, k=10)

# All metrics at once
metrics = compute_all_metrics(ranking, positive_indices, ks=[1, 5, 10])
```

### MTEB Evaluation

```python
from qwen3_rerank_trainer.evaluation import (
    set_proxy,
    MTEBRerankEvaluator,
    evaluate_reranking_dataset,
    evaluate_multiple_models,
)

# Set proxy (optional, call before importing mteb)
set_proxy("http://proxy:port")

# Evaluate with your reranker (with batch processing)
evaluator = MTEBRerankEvaluator(
    rerank_fn=my_rerank_fn,
    batch_size=50,   # max docs per request (avoid OOM)
    workers=8,       # concurrent requests
)
results = evaluator.evaluate("T2Reranking", max_samples=1000)

# Or evaluate multiple datasets
results = evaluator.evaluate_multiple(["chinese"])  # chinese, english, all

# Multi-model parallel evaluation
results = evaluate_multiple_models(
    rerankers={"model_a": reranker_a, "model_b": reranker_b},
    task_names=["chinese"],
    model_workers=2,  # 2 models evaluated in parallel
    batch_size=50,
)

# With GPU load balancing (avoid OOM on same GPU)
from qwen3_rerank_trainer.evaluation import evaluate_with_gpu_balance

results = evaluate_with_gpu_balance(
    rerankers={"9997": reranker_a, "9998": reranker_b, "10000": reranker_c},
    gpu_info={"9997": 0, "9998": 0, "10000": 1},  # model -> GPU mapping
    task_names=["chinese"],
    model_workers=2,  # distributed across GPUs
)
```

### API Reranker

```python
from qwen3_rerank_trainer.evaluation import (
    APIReranker,
    call_rerank_batch,
)

# Use APIReranker for evaluation
reranker = APIReranker(
    endpoint="http://localhost:9997/v1/rerank",
    model="Qwen3-Reranker-4B",
    batch_size=100,       # max docs per request (avoid OOM)
    max_concurrency=10,   # concurrent requests
)
ranking, scores = reranker.rerank(query, documents)

# Test connection
if reranker.test_connection():
    print("API is ready")

# Batch async rerank with progress bar
items = [(query1, docs1), (query2, docs2), ...]
results = call_rerank_batch(
    items,
    endpoint="http://localhost:9997/v1/rerank",
    max_concurrency=10,
    show_progress=True,
)
```

### Command Line Interface

```bash
# Install with eval support
pip install "qwen3-rerank-trainer[eval]"

# List supported datasets
qwen3-rerank-eval --list-datasets

# Evaluate single endpoint
qwen3-rerank-eval --endpoint http://localhost:9997 --datasets chinese

# Evaluate multiple endpoints (with GPU load balancing)
qwen3-rerank-eval --endpoints http://localhost:9997 http://localhost:9998 \
    --datasets chinese --model-workers 2

# Evaluate local dataset
qwen3-rerank-eval --endpoint http://localhost:9997 --input data.jsonl

# Dataset groups: chinese (4), english (6), multilingual (3), other (5), code (4), all (22)
```

### Two-Stage Evaluation (Embedding + Rerank)

```python
from qwen3_rerank_trainer.evaluation import run_two_stage_eval

results = run_two_stage_eval(
    embedding_config={"model_name": "...", "api_base": "..."},
    rerank_config={"model_name": "...", "api_base": "..."},
    tasks=["T2Retrieval", "MMarcoRetrieval"],
    output_dir="eval_output",
    proxy="http://proxy:port",
)
```

### Inference

```python
from qwen3_rerank_trainer import Qwen3Reranker

reranker = Qwen3Reranker("path/to/model")
ranked_docs = reranker.rerank(query, documents)
doc_scores = reranker.rerank(query, documents, return_scores=True)
```

### Data Processing

```python
from qwen3_rerank_trainer import (
    PREFIX, SUFFIX,
    format_input,
    sample_documents,
    tokenize_for_training,
)

# Format input for Qwen3-Reranker
text = format_input(query, document)

# Sample documents by difficulty
sampled_docs, sampled_labels = sample_documents(docs, n_total=10, n_pos=2)
```

## Package Structure

```
qwen3_rerank_trainer/
├── losses/              # Loss functions
│   ├── lambda_loss.py   # LambdaLoss + weighting schemes
│   ├── listwise.py      # ListMLE, p-ListMLE, ListNet
│   ├── pairwise.py      # RankNet, pairwise ranking
│   ├── pointwise.py     # BCE, CE
│   └── contrastive.py   # InfoNCE, multi-positive (mode switch)
├── training/            # Training utilities
│   ├── dataset.py       # RerankDataset (SFT)
│   ├── collator.py      # RerankCollator (SFT)
│   ├── sft_trainer.py   # ContrastiveSFTTrainer
│   ├── rl_dataset.py    # RLRerankDataset, RLCollator
│   ├── rl_trainer.py    # RLTrainer, load_sft_model
│   ├── cli.py           # SFT CLI
│   └── rl_cli.py        # RL CLI
├── rl/                  # RL losses and rewards
│   ├── rewards.py       # Doc-level rewards
│   └── losses.py        # REINFORCE, DPO
├── evaluation/          # Evaluation
│   ├── metrics.py       # MRR, AP, NDCG, P@k, R@k
│   ├── mteb_runner.py   # MTEB evaluation + multi-model parallel
│   ├── api_client.py    # Async API client + batch processing
│   ├── gpu_utils.py     # GPU load balancing
│   ├── two_stage_eval.py # Embedding + Rerank
│   └── report.py        # Report generation
├── inference/           # Inference
│   ├── base.py          # Base class
│   └── qwen_reranker.py # Qwen3-Reranker
└── data/                # Data processing
    ├── formatting.py    # Input formatting
    ├── sampling.py      # Document sampling
    └── tokenization.py  # Tokenization
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dullne/qwen3-rerank-trainer&type=Date)](https://www.star-history.com/#Dullne/qwen3-rerank-trainer&Date)
