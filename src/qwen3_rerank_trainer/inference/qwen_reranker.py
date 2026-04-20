"""
Qwen3-Reranker 推理类

与官方推理代码完全一致。
"""

import torch
from typing import List, Optional, Union

from .base import BaseReranker

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class Qwen3Reranker(BaseReranker):
    """Qwen3-Reranker 推理类"""

    def __init__(
        self,
        model_path: str,
        device: str = None,
        max_length: int = 8192,
        use_flash_attention: bool = False
    ):
        """
        初始化 Qwen3-Reranker

        Args:
            model_path: 模型路径
            device: 设备（cuda/cpu）
            max_length: 最大序列长度
            use_flash_attention: 是否使用 flash attention
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for Qwen3Reranker. "
                "Install with: pip install rerank-core[inference]"
            )

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            trust_remote_code=True
        )

        if use_flash_attention and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            ).cuda().eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                trust_remote_code=True
            ).to(self.device).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_instruction(self, instruction: Optional[str], query: str, doc: str) -> str:
        """格式化指令"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction,
            query=query,
            doc=doc
        )
        return output

    def process_inputs(self, pairs: List[str]):
        """处理输入"""
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)

        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs):
        """计算 logits 并返回分数"""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        return_scores: bool = False,
        batch_size: int = 32
    ):
        """
        重排序文档

        Args:
            query: 查询文本（支持批量）
            documents: 候选文档列表
            instruction: 任务指令
            top_k: 返回前k个文档
            return_scores: 是否返回分数
            batch_size: 批处理大小

        Returns:
            排序后的文档（和分数）
        """
        if isinstance(query, list):
            results = []
            for q in query:
                result = self.rerank(
                    q, documents, instruction, top_k, return_scores, batch_size
                )
                results.append(result)
            return results

        pairs = [
            self.format_instruction(instruction, query, doc)
            for doc in documents
        ]

        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = self.process_inputs(batch)
            scores = self.compute_logits(inputs)
            all_scores.extend(scores)

        doc_scores = list(zip(documents, all_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            doc_scores = doc_scores[:top_k]

        if return_scores:
            return doc_scores
        else:
            return [doc for doc, _ in doc_scores]
