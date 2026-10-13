from pathlib import Path
import json
import jieba
from rank_bm25 import BM25Okapi

project_dir = Path(__file__).resolve().parent
documentation_dir = project_dir / "data" / "chunks.json"
def tokenize(text):
    tokens = jieba.lcut(text)
    clean_tokens =[]
    
    for token in tokens:
        if token.isalnum():
            clean_tokens.append(token.lower())
    return clean_tokens
query = input("\n请输入问题：").strip()
query_tokens = tokenize(query)
print("查询词：", query_tokens)

chunks = json.loads(documentation_dir.read_text(encoding="utf-8"))
tokenized_chunks =[]
for chunk in chunks:
    tokenized_chunks.append(tokenize(chunk["text"]))

bm25 = BM25Okapi(tokenized_chunks)
scores = bm25.get_scores(query_tokens)
print("各片段的 BM25 分数：", scores)
results = []

for index,chunk in enumerate(chunks,start=1):
    document_tokens = tokenize(chunk["text"])
    matched_tokens = []
    
    for token in query_tokens:
        if token in document_tokens and token not in matched_tokens:
            matched_tokens.append(token)
    score = float(scores[index-1])
    results.append({
        "chunk_id": index,
        "score": score,
        "matched_tokens": matched_tokens,
        "chunk": chunk,
    })
results = sorted(results, key=lambda x: x["score"], reverse=True)
top_k = 2
matched_results =[]
for result in results:
    if result["matched_tokens"]:
        matched_results.append(result)

top_results = matched_results[:top_k]
if not top_results:
    print("没有找到匹配的片段。")

for rank, result in enumerate(top_results, start=1):
    chunk = result["chunk"]

    print(f"\n排名 {rank}：原文第 {result['chunk_id']} 块")
    print("分数：", result["score"])
    print("匹配词：", result["matched_tokens"])
    print(f"来源：{chunk['source']}")
    print(f"字符范围：[{chunk['start']}:{chunk['end']}]")
    print(chunk["text"])