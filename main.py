from pathlib import Path
import json

peoject_dir = Path(__file__).parent
documentation_dir = peoject_dir / "data" / "sample.txt"
text = documentation_dir.read_text(encoding="utf-8")

print("文档路径", documentation_dir)
print("文档内容", text)

chunk_size = 40
overlap = 10
step =chunk_size - overlap
chunks =[]

for  start in range(0, len(text), step):
    end = min(start + chunk_size, len(text))
    chunk = {
        "text": text[start:end],
        "source": documentation_dir.name,
        "start": start,
        "end": end
    }
    chunks.append(chunk)

print(f"\n共有 {len(text)} 个字符，被分割成 {len(chunks)} 个块。")

for index,chunk in enumerate(chunks,start=1):
    print(f"\n第 {index} 块:，来源{chunk['source']}")
    print(f"字符范围： {chunk['start']} - {chunk['end']}")
    print(chunk["text"])

output_path = documentation_dir.parent / "chunks.json"
json_txt = json.dumps(chunks, ensure_ascii=False, indent=2)
output_path.write_text(json_txt, encoding="utf-8")
print(f"\n分块结果已保存到 {output_path}")