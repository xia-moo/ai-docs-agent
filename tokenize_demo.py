import jieba

def tokenize(text):
    tokens = jieba.lcut(text)
    clean_tokens =[]
    
    for token in tokens:
        if token.isalnum():
            clean_tokens.append(token.lower())
    return clean_tokens
query = "RAG 如何生成答案？"
document = "把这些内容提供给语言模型生成答案。"

print("查询词：", tokenize(query))
print("文档词：", tokenize(document))
query_tokens = tokenize(query)
document_tokens = tokenize(document)
matched_tokens = []

for token in query_tokens:
    if token in document_tokens and token not in matched_tokens:
        matched_tokens.append(token)
score = len(matched_tokens)
print("匹配词：", matched_tokens)
print("匹配分数：", score)