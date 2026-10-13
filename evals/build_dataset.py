"""Resolve authored positive evidence spans before running retrieval measurements."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [
    (
        "lg-durable",
        "LangGraph 如何在运行失败后从中断位置恢复？",
        "langgraph.md",
        "automatically resuming from exactly where they left off",
    ),
    (
        "lg-human",
        "LangGraph 怎样让人检查并修改运行中的 agent state？",
        "langgraph.md",
        "inspecting and modifying agent state",
    ),
    (
        "lg-memory",
        "LangGraph 的短期和长期记忆分别支持什么？",
        "langgraph.md",
        "long-term persistent memory across sessions",
    ),
    (
        "qd-payload",
        "Qdrant 中的 point 除了 vector 还可以包含什么？",
        "qdrant.md",
        "vectors with an additional payload",
    ),
    (
        "qd-filter",
        "Qdrant payload filtering 可以组合哪些条件子句？",
        "qdrant.md",
        "`should`, `must`, and `must_not`",
    ),
    (
        "qd-fusion",
        "Qdrant hybrid search 支持 RRF 和哪些融合策略？",
        "qdrant.md",
        "Reciprocal Rank Fusion (RRF)",
    ),
    ("bge-length", "BGE-M3 的输入长度最多支持多少 tokens？", "flagembedding.md", "input length up to 8192"),
    (
        "bge-modes",
        "BGE-M3 支持哪些 dense、lexical 和多向量检索方式？",
        "flagembedding.md",
        "unification of dense, lexical, multi-vec/colbert retrieval",
    ),
    (
        "bge-reranker",
        "bge-reranker-base 应该用于检索流程的哪个步骤？",
        "flagembedding.md",
        "re-rank top-k documents returned by embedding models",
    ),
    (
        "hf-ids",
        "Tokenizer 返回的 input_ids 表示什么？",
        "preprocessing_zh.md",
        "是与句子中每个`token`对应的索引",
    ),
    ("hf-mask", "attention_mask 的用途是什么？", "preprocessing_zh.md", "指示是否应该关注一个`token`"),
    (
        "hf-padding",
        "padding=True 如何处理一个批次里不同长度的句子？",
        "preprocessing_zh.md",
        "批次中较短的序列填充到与最长序列相匹配的长度",
    ),
    (
        "hf-truncation",
        "truncation=True 会怎样处理超过模型长度的输入？",
        "preprocessing_zh.md",
        "将序列截断为模型接受的最大长度",
    ),
    ("hf-tensors", "return_tensors 的 pt 和 tf 分别对应什么框架？", "preprocessing_zh.md", "设置为 `pt`"),
    (
        "hf-decode",
        "怎样把 input_ids 解码回原始文本？",
        "preprocessing_zh.md",
        'tokenizer.decode(encoded_input["input_ids"])',
    ),
    (
        "pipe-device",
        "pipeline 中 device=n 的作用是什么？",
        "pipeline_zh.md",
        "`pipeline`会自动将模型放在指定的设备上",
    ),
    (
        "pipe-accelerate",
        "使用 device_map=auto 需要安装什么依赖？",
        "pipeline_zh.md",
        "使用 `device_map` 参数需要安装",
    ),
    ("pipe-batch", "怎样给 pipeline 配置 batch_size 为 2？", "pipeline_zh.md", "batch_size=2"),
    (
        "chat-format",
        "消息列表应该交给 tokenizer 的哪个方法应用聊天模板？",
        "chat_template_zh.md",
        "将其传递给[`~PreTrainedTokenizer.apply_chat_template`]方法",
    ),
    (
        "chat-prompt",
        "聊天模型预测时如何添加 generation prompt？",
        "chat_template_zh.md",
        "添加[generation prompt]",
    ),
    (
        "chat-training",
        "训练时 add_generation_prompt 应该设置为 True 还是 False？",
        "chat_template_zh.md",
        "通常应该设置`add_generation_prompt=False`",
    ),
    (
        "chat-custom",
        "如何替换 tokenizer 使用的 chat_template？",
        "chat_template_zh.md",
        "设置`tokenizer.chat_template`属性",
    ),
    (
        "paper-architecture",
        "Transformer 的架构是否需要 recurrence 和 convolutions？",
        "attention_is_all_you_need.pdf",
        "dispensing with recurrence and convolutions entirely",
    ),
    (
        "paper-heads",
        "multi-head attention 为什么使用不同的 representation subspaces？",
        "attention_is_all_you_need.pdf",
        "jointly attend to information",
    ),
]


def normalize(text):
    return re.sub(r"\s+", "", text).lower()


def build():
    bundle = json.loads((ROOT / "data" / "index" / "index.json").read_text(encoding="utf-8"))
    questions = []
    for identifier, question, source, quote in SEEDS:
        positive = [
            chunk
            for chunk in bundle["chunks"]
            if chunk["source"] == source and normalize(quote) in normalize(chunk["text"])
        ]
        if not positive:
            raise ValueError(f"No chunk contains the authored evidence: {identifier}: {quote}")
        questions.append(
            {
                "id": identifier,
                "question": question,
                "source": source,
                "evidence_quote": quote,
                "relevant_ids": [chunk["id"] for chunk in positive],
                "locations": [
                    {"page": chunk["page"], "start": chunk["start"], "end": chunk["end"]}
                    for chunk in positive
                ],
            }
        )
    result = {
        "index_revision": bundle["revision"],
        "questions": questions,
        "labeling": "Authored source-span positives resolved before retrieval measurement; development set, not independently adjudicated.",
    }
    (ROOT / "evals" / "dataset.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Labelled", len(questions), "questions")


if __name__ == "__main__":
    build()
