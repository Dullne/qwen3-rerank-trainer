# qwen3-rerank-trainer

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://github.com/Dullne/qwen3-rerank-trainer/stargazers"><img src="https://img.shields.io/github/stars/Dullne/qwen3-rerank-trainer?style=social" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Dullne/qwen3-rerank-trainer" alt="License"></a>
</p>

面向 Qwen3-Reranker 的训练与评测工具包。

## 安装

```bash
# 基础安装
pip install -e .

# 安装推理支持
pip install -e ".[inference]"

# 安装 MTEB 评测支持
pip install -e ".[eval]"

# 安装两阶段评测支持（Embedding + Rerank）
pip install -e ".[evalscope]"

# 完整安装
pip install -e ".[full]"
```

## 功能特性

### 损失函数

```python
from qwen3_rerank_trainer import (
    lambda_loss,           # LambdaLoss 框架
    list_mle,              # ListMLE
    infonce_loss,          # InfoNCE
    ranknet_loss,          # RankNet
)
from qwen3_rerank_trainer.losses import NDCGLoss2PPScheme

# 使用 NDCG 优化的 LambdaLoss
loss = lambda_loss(scores, labels, metric="ndcg")
```

说明：`lambda_rank_loss` 和 `lambda_loss_ndcg/map/mrr` 已合并到 `lambda_loss`。

### SFT 训练

```bash
# 命令行（需要 pip install -e ".[full]"）
qwen3-rerank-train --model /path/to/Qwen3-Reranker-4B --data train.jsonl --output outputs/sft

# 使用 LoRA
qwen3-rerank-train --model /path/to/model --data train.jsonl --output outputs/sft \
    --lora --lora-r 8 --lora-alpha 16 --n-docs 8 --n-pos 1

# 不同损失函数
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type infonce --temperature 0.05
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type infonce --infonce-mode posset
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type lambda_loss --lambda-metric ndcg
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type list_mle
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type ranknet
qwen3-rerank-train --model /path/to/model --data train.jsonl --loss-type ranknet --ranknet-max-pairs 2000000

# 可选：过滤超长样本（默认关闭）
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

# 加载数据集
dataset = RerankDataset(
    "train.jsonl",
    tokenizer=tokenizer,
    n_docs=8,
    n_pos=1,  # 固定 1 正 7 负
)

# 大数据集可使用流式版本，避免一次性占用过多内存
# dataset = StreamingRerankDataset("train.jsonl", tokenizer=tokenizer, n_docs=8, n_pos=1)

# 创建 collator
collator = RerankCollator(tokenizer, max_length=4096)

# 获取 yes/no token ID
yes_id, no_id = get_yes_no_token_ids(tokenizer)

# 创建 trainer
trainer = ContrastiveSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    yes_token_id=yes_id,
    no_token_id=no_id,
    chunk_size=16,  # 分块处理，节省显存
)
trainer.train()
```

### RL 训练

```bash
# 命令行（需要 pip install -e ".[full]"）
# 基础 RL 训练（需要先完成 SFT 训练）
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --output outputs/rl

# 使用所有文档（推荐用于大规模数据）
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl \
    --n_docs 0 --max_docs 50  # 使用所有文档，但每个样本最多保留 50 个

# 分块前向传播（节省显存，支持任意大的 n_docs）
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl \
    --chunk_size 8  # 每次处理 8 个文档

# DAPO 损失（默认）
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dapo

# Dr. GRPO 损失
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dr_grpo

# DPO 损失
qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dpo --dpo_beta 0.1

# 可选：过滤超长样本（默认关闭）
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

# 加载 SFT 模型
model = load_sft_model("outputs/sft/final", "Qwen/Qwen3-Reranker-4B")

# 添加新的 LoRA adapter
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)

# 准备数据集
dataset = RLRerankDataset(
    "train.jsonl",
    tokenizer=tokenizer,
    n_docs=8,       # 每组 8 个文档
    n_pos=0,        # 按原始比例动态分配
    max_docs=50,    # 限制极端样本的文档数
)

# 大数据集可使用流式版本
# dataset = StreamingRLRerankDataset("train.jsonl", tokenizer=tokenizer, n_docs=8, n_pos=0, max_docs=50)

# 创建 collator
collator = RLCollator(tokenizer, max_length=4096)

# 创建 trainer
trainer = RLTrainer(
    yes_token_id=tokenizer.convert_tokens_to_ids("yes"),
    no_token_id=tokenizer.convert_tokens_to_ids("no"),
    kl_coef=0.1,           # KL 惩罚系数
    reward_type="rank_based",  # rank_based, ndcg_based, recall_based
    loss_type="dapo",      # grpo, dapo, dr_grpo
    chunk_size=8,          # 分块处理，节省显存
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

# 根据每个文档的分数和标签计算 rewards / advantages
rewards = compute_doc_level_rewards(scores, labels, reward_type="ndcg_based")
advantages = compute_doc_level_advantages(scores, labels, reward_type="ndcg_based")

# 根据 yes/no logits 计算 REINFORCE loss
loss, advantages, rewards, kl = reinforce_loss(yes_logits, no_logits, labels)
```

### 评测

```python
from qwen3_rerank_trainer import mrr, ndcg_at_k, compute_all_metrics

# 基础指标
mrr_score = mrr(ranking, positive_indices)
ndcg_score = ndcg_at_k(ranking, relevance_scores, k=10)

# 一次性计算全部指标
metrics = compute_all_metrics(ranking, positive_indices, ks=[1, 5, 10])
```

### MTEB 评测

```python
from qwen3_rerank_trainer.evaluation import (
    set_proxy,
    MTEBRerankEvaluator,
    evaluate_reranking_dataset,
    evaluate_multiple_models,
)

# 设置代理（可选，需要在导入 mteb 前调用）
set_proxy("http://proxy:port")

# 使用自定义 reranker 评测（支持批处理）
evaluator = MTEBRerankEvaluator(
    rerank_fn=my_rerank_fn,
    batch_size=50,   # 每次请求的最大文档数，避免 OOM
    workers=8,       # 并发请求数
)
results = evaluator.evaluate("T2Reranking", max_samples=1000)

# 或一次评测多个数据集
results = evaluator.evaluate_multiple(["chinese"])  # chinese, english, all

# 多模型并行评测
results = evaluate_multiple_models(
    rerankers={"model_a": reranker_a, "model_b": reranker_b},
    task_names=["chinese"],
    model_workers=2,  # 2 个模型并行评测
    batch_size=50,
)

# GPU 负载均衡，避免同一张卡 OOM
from qwen3_rerank_trainer.evaluation import evaluate_with_gpu_balance

results = evaluate_with_gpu_balance(
    rerankers={"9997": reranker_a, "9998": reranker_b, "10000": reranker_c},
    gpu_info={"9997": 0, "9998": 0, "10000": 1},  # model -> GPU 映射
    task_names=["chinese"],
    model_workers=2,  # 分散到不同 GPU
)
```

### API Reranker

```python
from qwen3_rerank_trainer.evaluation import (
    APIReranker,
    call_rerank_batch,
)

# 使用 APIReranker 做评测
reranker = APIReranker(
    endpoint="http://localhost:9997/v1/rerank",
    model="Qwen3-Reranker-4B",
    batch_size=100,       # 每次请求的最大文档数，避免 OOM
    max_concurrency=10,   # 并发请求数
)
ranking, scores = reranker.rerank(query, documents)

# 测试连接
if reranker.test_connection():
    print("API is ready")

# 带进度条的批量异步 rerank
items = [(query1, docs1), (query2, docs2), ...]
results = call_rerank_batch(
    items,
    endpoint="http://localhost:9997/v1/rerank",
    max_concurrency=10,
    show_progress=True,
)
```

### 命令行评测

```bash
# 安装 eval 依赖
pip install -e ".[eval]"

# 列出支持的数据集
qwen3-rerank-eval --list-datasets

# 评测单个 endpoint
qwen3-rerank-eval --endpoint http://localhost:9997 --datasets chinese

# 评测多个 endpoints（支持 GPU 负载均衡）
qwen3-rerank-eval --endpoints http://localhost:9997 http://localhost:9998 \
    --datasets chinese --model-workers 2

# 评测本地数据集
qwen3-rerank-eval --endpoint http://localhost:9997 --input data.jsonl

# 数据集分组：chinese (4), english (6), multilingual (3), other (5), code (4), all (22)
```

### 两阶段评测（Embedding + Rerank）

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

### 推理

```python
from qwen3_rerank_trainer import Qwen3Reranker

reranker = Qwen3Reranker("path/to/model")
ranked_docs = reranker.rerank(query, documents)
doc_scores = reranker.rerank(query, documents, return_scores=True)
```

### 数据处理

```python
from qwen3_rerank_trainer import (
    PREFIX, SUFFIX,
    format_input,
    sample_documents,
    tokenize_for_training,
)

# 格式化 Qwen3-Reranker 输入
text = format_input(query, document)

# 按难度采样文档
sampled_docs, sampled_labels = sample_documents(docs, n_total=10, n_pos=2)
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dullne/qwen3-rerank-trainer&type=Date)](https://www.star-history.com/#Dullne/qwen3-rerank-trainer&Date)

## 包结构

```
qwen3_rerank_trainer/
├── losses/              # 损失函数
│   ├── lambda_loss.py   # LambdaLoss + weighting schemes
│   ├── listwise.py      # ListMLE, p-ListMLE, ListNet
│   ├── pairwise.py      # RankNet, pairwise ranking
│   ├── pointwise.py     # BCE, CE
│   └── contrastive.py   # InfoNCE, multi-positive (mode switch)
├── training/            # 训练工具
│   ├── dataset.py       # RerankDataset (SFT)
│   ├── collator.py      # RerankCollator (SFT)
│   ├── sft_trainer.py   # ContrastiveSFTTrainer
│   ├── rl_dataset.py    # RLRerankDataset, RLCollator
│   ├── rl_trainer.py    # RLTrainer, load_sft_model
│   ├── cli.py           # SFT CLI
│   └── rl_cli.py        # RL CLI
├── rl/                  # RL 损失和奖励
│   ├── rewards.py       # Doc-level rewards
│   └── losses.py        # REINFORCE, DPO
├── evaluation/          # 评测
│   ├── metrics.py       # MRR, AP, NDCG, P@k, R@k
│   ├── mteb_runner.py   # MTEB evaluation + multi-model parallel
│   ├── api_client.py    # Async API client + batch processing
│   ├── gpu_utils.py     # GPU load balancing
│   ├── two_stage_eval.py # Embedding + Rerank
│   └── report.py        # Report generation
├── inference/           # 推理
│   ├── base.py          # Base class
│   └── qwen_reranker.py # Qwen3-Reranker
└── data/                # 数据处理
    ├── formatting.py    # Input formatting
    ├── sampling.py      # Document sampling
    └── tokenization.py  # Tokenization
```

## 许可证

MIT
