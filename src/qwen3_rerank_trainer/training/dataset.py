"""
Rerank Dataset for SFT Training

支持多种数据格式的重排序数据集。
"""
import json
import random
import logging
from itertools import islice
from pathlib import Path
from typing import List, Dict, Optional

from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ..data.formatting import PREFIX, SUFFIX, format_input

logger = logging.getLogger(__name__)


def _load_hf_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset as hf_load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Loading this data format requires a working 'datasets' installation"
        ) from exc
    return hf_load_dataset(*args, **kwargs)


LIST_FIELD_KEYS = [
    "positives",
    "negatives",
    "positive",
    "negative",
    "pos",
    "positives_strong",
    "positives_medium",
    "positives_weak",
    "neg_very_hard",
    "neg_hard",
    "neg_medium",
    "statement_very_hard_negatives",
    "statement_hard_negatives",
    "statement_medium_negatives",
]


def _parse_list_field(value):
    """尝试将字符串解析为列表（处理 CSV 加载的情况）。

    支持格式：
    - JSON: '["pos1", "pos2"]'
    - Python: "['pos1', 'pos2']"
    - NumPy/HF: "['pos1' 'pos2']" (空格分隔，自动转换)
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value

    import json
    import ast
    import re

    value = value.strip()

    # 1. 尝试 JSON 解析
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. 检测并处理 NumPy/HF 风格: ['pos1' 'pos2'] (引号间有空格无逗号)
    #    必须在 ast.literal_eval 之前处理，否则 Python 会拼接相邻字符串
    if value.startswith("[") and value.endswith("]"):
        if re.search(r"'\s+'", value) or re.search(r'"\s+"', value):
            converted = re.sub(r"'\s+'", "', '", value)
            converted = re.sub(r'"\s+"', '", "', converted)
            try:
                parsed = ast.literal_eval(converted)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass

    # 3. 尝试 Python literal 解析
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return value


def _normalize_columns(row: Dict) -> Dict:
    """规范化列名和字段类型。

    1. 将单数形式的列名转换为复数形式：
       - positive → positives
       - negative → negatives
    2. 尝试将字符串形式的列表解析为真正的列表（CSV 兼容）
    """
    if "positive" in row and "positives" not in row:
        row["positives"] = row.pop("positive")
    if "negative" in row and "negatives" not in row:
        row["negatives"] = row.pop("negative")

    # 处理 CSV 加载时列表变字符串的问题
    for key in LIST_FIELD_KEYS:
        if key in row:
            row[key] = _parse_list_field(row[key])

    return row


def _iter_data_for_worker(data_path: str):
    worker_info = get_worker_info()
    if worker_info is None:
        yield from iter_data(data_path)
        return

    yield from islice(iter_data(data_path), worker_info.id, None, worker_info.num_workers)


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value, key=str)
    return [value]


def _clean_text_list(values) -> List[str]:
    cleaned = []
    for value in _as_list(values):
        if isinstance(value, dict):
            value = value.get("statement", "") or value.get("text", "")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def load_data(data_path: str) -> List[Dict]:
    """加载本地数据文件，支持多种格式。

    支持格式: jsonl, json, parquet, csv, arrow

    自动规范化列名：positive → positives, negative → negatives

    Args:
        data_path: 本地数据文件或目录路径

    Returns:
        数据记录列表
    """
    path = Path(data_path)

    # 目录：找第一个数据文件
    if path.is_dir():
        for pattern in ["*.jsonl", "*.json", "*.parquet", "*.csv", "*.arrow"]:
            files = sorted(path.glob(pattern))
            if files:
                path = files[0]
                logger.info(f"从目录中选择文件: {path}")
                break
        else:
            raise FileNotFoundError(f"目录中未找到数据文件: {data_path}")

    # 根据后缀确定格式
    suffix = path.suffix.lower()
    format_map = {
        ".jsonl": "json",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
        ".arrow": "arrow",
    }

    if suffix not in format_map:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: {list(format_map.keys())}")

    logger.info(f"加载数据文件: {path} (格式: {format_map[suffix]})")

    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return [_normalize_columns(dict(row)) for row in records]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            data = data["data"]
        elif isinstance(data, dict):
            data = [data]
        return [_normalize_columns(dict(row)) for row in data]

    ds = _load_hf_dataset(format_map[suffix], data_files=str(path), split="train")
    return [_normalize_columns(dict(row)) for row in ds]


def iter_data(data_path: str):
    """流式加载本地数据文件，支持多种格式。

    支持格式: jsonl, json, parquet, csv, arrow

    Args:
        data_path: 本地数据文件或目录路径

    Yields:
        规范化后的数据记录
    """
    path = Path(data_path)

    # 目录：找第一个数据文件
    if path.is_dir():
        for pattern in ["*.jsonl", "*.json", "*.parquet", "*.csv", "*.arrow"]:
            files = sorted(path.glob(pattern))
            if files:
                path = files[0]
                logger.info(f"从目录中选择文件: {path}")
                break
        else:
            raise FileNotFoundError(f"目录中未找到数据文件: {data_path}")

    suffix = path.suffix.lower()
    format_map = {
        ".jsonl": "json",
        ".json": "json",
        ".parquet": "parquet",
        ".csv": "csv",
        ".arrow": "arrow",
    }

    if suffix not in format_map:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: {list(format_map.keys())}")

    logger.info(f"流式加载数据文件: {path} (格式: {format_map[suffix]})")

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield _normalize_columns(json.loads(line))
        return

    if suffix == ".json":
        for row in load_data(str(path)):
            yield row
        return

    ds = _load_hf_dataset(format_map[suffix], data_files=str(path), split="train", streaming=True)
    for row in ds:
        yield _normalize_columns(dict(row))


class RerankDataset(Dataset):
    """重排序数据集

    支持多种数据格式：
    - 简单格式: {query, positives, negatives}
    - 分级格式: {query, pos, neg_very_hard, neg_hard, neg_medium}

    正负例采样策略：
    - n_docs = 0: 使用所有文档（不限制数量，仅受 min_pos/min_neg 约束）
    - n_pos > 0: 固定正例数（n_pos 正，n_docs - n_pos 负）
    - n_pos = 0: 动态分配（按原始比例，min_pos/min_neg 约束）

    Args:
        data_file: JSONL 数据文件路径
        tokenizer: 可选 tokenizer（仅在 filter_overlength=True 时用于长度过滤）
        max_length: 最大 token 长度，超过的文档会被过滤
        n_docs: 每组文档数（0 表示使用所有文档）
        n_pos: 固定正例数（0 表示按原始比例动态分配）
        min_pos: 最少正例数（仅动态分配时生效）
        min_neg: 最少负例数（仅动态分配时生效）
        seed: 随机种子
        max_samples: 最大样本数（0 表示不限制）
        format_fn: 自定义格式化函数（用于长度检查）
        filter_overlength: 是否过滤超过 max_length 的样本（默认关闭）

    Example:
        >>> dataset = RerankDataset(
        ...     "train.jsonl",
        ...     tokenizer=tokenizer,
        ...     n_docs=8,
        ...     n_pos=1,  # 固定 1 正 7 负
        ... )
        >>> item = dataset[0]
        >>> print(item.keys())  # dict_keys(['query', 'positives', 'negatives'])
    """

    def __init__(
        self,
        data_file: str,
        tokenizer=None,
        max_length: int = 4096,
        n_docs: int = 8,
        n_pos: int = 0,
        min_pos: int = 1,
        min_neg: int = 1,
        seed: int = 42,
        max_samples: int = 0,
        format_fn: Optional[callable] = None,
        filter_overlength: bool = False,
    ):
        self.n_docs = n_docs
        self.n_pos = n_pos
        self.min_pos = min_pos
        self.min_neg = min_neg
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.format_fn = format_fn or self._default_format
        self.filter_overlength = filter_overlength
        self.seed = seed
        self.rng = random.Random(seed)
        self._worker_rng = None
        self._worker_id = None

        # 加载数据
        raw_data = load_data(data_file)
        logger.info(f"加载 {len(raw_data)} 条记录")

        # 检测数据格式
        sample = raw_data[0] if raw_data else {}
        is_simple_format = 'positives' in sample and 'negatives' in sample
        is_graded_format = 'neg_very_hard' in sample or 'neg_hard' in sample

        if is_simple_format:
            logger.info("数据格式: 简单格式 (positives/negatives)")
        elif is_graded_format:
            logger.info("数据格式: 分级格式 (neg_very_hard/hard/medium)")
        else:
            logger.info("数据格式: 旧格式 (pos/statement_*_negatives)")

        # 处理数据
        self.data = []
        for item in raw_data:
            query = item.get("query", "")
            if not query:
                continue

            # 提取正例
            positives = self._extract_positives(item, is_simple_format)
            # 提取负例
            negatives = self._extract_negatives(item, is_simple_format)

            # 过滤超长文档
            if self.tokenizer is not None and self.filter_overlength:
                positives = [p for p in positives if self._check_length(query, p)]
                negatives = [n for n in negatives if self._check_length(query, n)]

            # 确保有足够的正例和负例
            if not self._has_enough_docs(positives, negatives):
                continue

            self.data.append({
                "query": query,
                "positives": positives,
                "negatives": negatives
            })

        logger.info(f"有效样本: {len(self.data)}")
        if self.tokenizer is not None and self.filter_overlength:
            logger.info(f"已过滤超过 {self.max_length} tokens 的文档")

        # 限制样本数量
        if max_samples > 0 and len(self.data) > max_samples:
            self.rng.shuffle(self.data)
            self.data = self.data[:max_samples]
            logger.info(f"限制样本数: {max_samples}")

    def _extract_positives(self, item: Dict, is_simple_format: bool) -> List[str]:
        """提取正例"""
        if is_simple_format:
            return _clean_text_list(item.get("positives", []))

        positives = []
        if item.get("answer"):
            positives.append(item["answer"])
        positives.extend(_clean_text_list(item.get("pos", [])))
        positives.extend(_clean_text_list(item.get("positives_strong", [])))
        positives.extend(_clean_text_list(item.get("positives_medium", [])))
        positives.extend(_clean_text_list(item.get("positives_weak", [])))
        return _clean_text_list(positives)

    def _extract_negatives(self, item: Dict, is_simple_format: bool) -> List[str]:
        """提取负例"""
        if is_simple_format:
            return _clean_text_list(item.get("negatives", []))

        negatives = []
        for key in ["neg_very_hard", "neg_hard", "neg_medium",
                   "statement_very_hard_negatives", "statement_hard_negatives", "statement_medium_negatives"]:
            negatives.extend(_clean_text_list(item.get(key, [])))
        return negatives

    def _has_enough_docs(self, positives: List[str], negatives: List[str]) -> bool:
        """检查是否有足够的文档"""
        if self.n_docs == 0:
            # 使用所有文档模式：只检查最小值约束
            return len(positives) >= self.min_pos and len(negatives) >= self.min_neg
        elif self.n_pos > 0:
            # 固定正例数模式
            required_neg = self.n_docs - self.n_pos
            return len(positives) >= self.n_pos and len(negatives) >= required_neg
        else:
            # 动态分配模式
            if len(positives) < self.min_pos or len(negatives) < self.min_neg:
                return False
            return len(positives) + len(negatives) >= self.n_docs

    def _default_format(self, query: str, document: str) -> str:
        """默认格式化（Qwen3-Reranker 格式）"""
        return f"{PREFIX}{format_input(query, document)}{SUFFIX}"

    def _check_length(self, query: str, document: str) -> bool:
        """检查文本长度是否超过 max_length"""
        text = self.format_fn(query, document)
        tokens = self.tokenizer(text, add_special_tokens=False)
        return len(tokens["input_ids"]) <= self.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]

        # 采样文档
        positives = item["positives"].copy()
        negatives = item["negatives"].copy()

        rng = self._get_rng()
        rng.shuffle(positives)
        rng.shuffle(negatives)

        if self.n_docs == 0:
            # 使用所有文档模式
            n_pos = len(positives)
            n_neg = len(negatives)
        elif self.n_pos > 0:
            # 固定正例数模式
            n_pos = self.n_pos
            n_neg = self.n_docs - self.n_pos
        else:
            # 动态分配模式
            total_available = len(positives) + len(negatives)
            pos_ratio = len(positives) / total_available

            n_pos = max(self.min_pos, round(self.n_docs * pos_ratio))
            n_neg = self.n_docs - n_pos

            if n_neg < self.min_neg:
                n_neg = self.min_neg
                n_pos = self.n_docs - n_neg

            n_pos = min(n_pos, len(positives))
            n_neg = min(n_neg, len(negatives))

            if n_pos + n_neg < self.n_docs:
                n_neg = min(self.n_docs - n_pos, len(negatives))
                n_pos = min(self.n_docs - n_neg, len(positives))

        # 选择正例和负例
        selected_pos = positives[:n_pos]
        selected_neg = negatives[:n_neg]

        return {
            "query": query,
            "positives": selected_pos,
            "negatives": selected_neg
        }

    def _get_rng(self) -> random.Random:
        """为多 worker 提供独立 RNG，避免采样重复。"""
        worker_info = get_worker_info()
        if worker_info is None:
            return self.rng
        if self._worker_rng is None or self._worker_id != worker_info.id:
            self._worker_id = worker_info.id
            self._worker_rng = random.Random(worker_info.seed)
        return self._worker_rng


class StreamingRerankDataset(IterableDataset):
    """流式版本的 RerankDataset（用于大数据集，避免内存占用）。

    参数与 RerankDataset 保持一致，但不支持 __len__。
    """

    def __init__(
        self,
        data_file: str,
        tokenizer=None,
        max_length: int = 4096,
        n_docs: int = 8,
        n_pos: int = 0,
        min_pos: int = 1,
        min_neg: int = 1,
        seed: int = 42,
        max_samples: int = 0,
        format_fn: Optional[callable] = None,
        filter_overlength: bool = False,
    ):
        self.data_file = data_file
        self.n_docs = n_docs
        self.n_pos = n_pos
        self.min_pos = min_pos
        self.min_neg = min_neg
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.format_fn = format_fn or self._default_format
        self.filter_overlength = filter_overlength
        self.seed = seed
        self.rng = random.Random(seed)
        self._worker_rng = None
        self._worker_id = None
        self.max_samples = max_samples

    def __iter__(self):
        rng = self._get_rng()
        count = 0
        for item in _iter_data_for_worker(self.data_file):
            query = item.get("query", "")
            if not query:
                continue

            sample = item if item else {}
            is_simple_format = 'positives' in sample and 'negatives' in sample

            positives = self._extract_positives(item, is_simple_format)
            negatives = self._extract_negatives(item, is_simple_format)

            if self.tokenizer is not None and self.filter_overlength:
                positives = [p for p in positives if self._check_length(query, p)]
                negatives = [n for n in negatives if self._check_length(query, n)]

            if not self._has_enough_docs(positives, negatives):
                continue

            rng.shuffle(positives)
            rng.shuffle(negatives)

            if self.n_docs == 0:
                n_pos = len(positives)
                n_neg = len(negatives)
            elif self.n_pos > 0:
                n_pos = self.n_pos
                n_neg = self.n_docs - self.n_pos
            else:
                total_available = len(positives) + len(negatives)
                pos_ratio = len(positives) / total_available

                n_pos = max(self.min_pos, round(self.n_docs * pos_ratio))
                n_neg = self.n_docs - n_pos

                if n_neg < self.min_neg:
                    n_neg = self.min_neg
                    n_pos = self.n_docs - n_neg

                n_pos = min(n_pos, len(positives))
                n_neg = min(n_neg, len(negatives))

                if n_pos + n_neg < self.n_docs:
                    n_neg = min(self.n_docs - n_pos, len(negatives))
                    n_pos = min(self.n_docs - n_neg, len(positives))

            selected_pos = positives[:n_pos]
            selected_neg = negatives[:n_neg]

            yield {
                "query": query,
                "positives": selected_pos,
                "negatives": selected_neg,
            }

            count += 1
            if self.max_samples > 0 and count >= self.max_samples:
                break

    def _extract_positives(self, item: Dict, is_simple_format: bool) -> List[str]:
        if is_simple_format:
            return _clean_text_list(item.get("positives", []))
        positives = []
        if item.get("answer"):
            positives.append(item["answer"])
        positives.extend(_clean_text_list(item.get("pos", [])))
        positives.extend(_clean_text_list(item.get("positives_strong", [])))
        positives.extend(_clean_text_list(item.get("positives_medium", [])))
        positives.extend(_clean_text_list(item.get("positives_weak", [])))
        return _clean_text_list(positives)

    def _extract_negatives(self, item: Dict, is_simple_format: bool) -> List[str]:
        if is_simple_format:
            return _clean_text_list(item.get("negatives", []))
        negatives = []
        for key in ["neg_very_hard", "neg_hard", "neg_medium",
                   "statement_very_hard_negatives", "statement_hard_negatives", "statement_medium_negatives"]:
            negatives.extend(_clean_text_list(item.get(key, [])))
        return negatives

    def _has_enough_docs(self, positives: List[str], negatives: List[str]) -> bool:
        if self.n_docs == 0:
            return len(positives) >= self.min_pos and len(negatives) >= self.min_neg
        elif self.n_pos > 0:
            required_neg = self.n_docs - self.n_pos
            return len(positives) >= self.n_pos and len(negatives) >= required_neg
        else:
            if len(positives) < self.min_pos or len(negatives) < self.min_neg:
                return False
            return len(positives) + len(negatives) >= self.n_docs

    def _default_format(self, query: str, document: str) -> str:
        return f"{PREFIX}{format_input(query, document)}{SUFFIX}"

    def _check_length(self, query: str, document: str) -> bool:
        text = self.format_fn(query, document)
        tokens = self.tokenizer(text, add_special_tokens=False)
        return len(tokens["input_ids"]) <= self.max_length

    def _get_rng(self) -> random.Random:
        worker_info = get_worker_info()
        if worker_info is None:
            return self.rng
        if self._worker_rng is None or self._worker_id != worker_info.id:
            self._worker_id = worker_info.id
            self._worker_rng = random.Random(worker_info.seed)
        return self._worker_rng
