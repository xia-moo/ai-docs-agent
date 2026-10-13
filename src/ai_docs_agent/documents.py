"""Parse plain text, Markdown sections and PDF pages with original source offsets."""

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

ALLOWED_SUFFIXES = {".txt", ".md", ".pdf"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    section: str
    page: int | None
    line_start: int | None
    line_end: int | None
    start: int
    end: int
    source_url: str | None = None

    def to_dict(self):
        return asdict(self)


def _sections(text: str, markdown: bool):
    if not markdown:
        return [(0, len(text), "正文")]
    sections, headings = [], []
    offset, start, title, fence = 0, 0, "前言", None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        delimiter = re.match(r"^(`{3,}|~{3,})", stripped)
        if delimiter:
            char = delimiter.group(1)[0]
            if fence is None:
                fence = char
            elif fence == char:
                fence = None
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if heading:
            if offset > start and text[start:offset].strip():
                sections.append((start, offset, title))
            depth = len(heading.group(1))
            while headings and headings[-1][0] >= depth:
                headings.pop()
            headings.append((depth, heading.group(2).strip()))
            title = " / ".join(value for _, value in headings)
            start = offset  # keep the heading in the chunk for technical context
        offset += len(line)
    if start < len(text) and text[start:].strip():
        sections.append((start, len(text), title))
    return sections


def parse_document(path: Path, chunk_size=350, overlap=60, source_url=None) -> list[Chunk]:
    if not (50 <= chunk_size <= 1500) or not (0 <= overlap < chunk_size):
        raise ValueError("切分参数必须满足 50≤chunk_size≤1500、0≤overlap<chunk_size。")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("支持 TXT、Markdown 和含文字的 PDF。")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("单个文件不能超过 10 MB。")
    pages = []
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError("请先解除 PDF 密码保护。")
            if len(reader.pages) > 500:
                raise ValueError("单个 PDF 最多支持 500 页。")
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF 无法解析，请检查文件是否完整。") from exc
    else:
        try:
            pages = [(None, path.read_text(encoding="utf-8-sig"))]
        except UnicodeError as exc:
            raise ValueError("TXT/Markdown 请保存为 UTF-8 编码。") from exc
    if sum(len(text) for _, text in pages) > MAX_TEXT_CHARS:
        raise ValueError("解析文本过长，请拆分文档。")
    chunks = []
    for page, text in pages:
        for begin, finish, section in _sections(text, path.suffix.lower() == ".md"):
            start = begin
            while start < finish:
                end = min(start + chunk_size, finish)
                # Prefer a complete paragraph/sentence when the window has enough content.
                if end < finish:
                    tail = text[start + chunk_size // 2 : end]
                    boundary = max(tail.rfind("\n"), tail.rfind("。"), tail.rfind(". "))
                    if boundary >= 0:
                        end = start + chunk_size // 2 + boundary + 1
                fragment = text[start:end]
                if fragment.strip():
                    key = f"{path.name}|{page}|{section}|{start}|{end}|{fragment}"
                    chunks.append(
                        Chunk(
                            id=hashlib.sha256(key.encode()).hexdigest()[:32],
                            text=fragment,
                            source=path.name,
                            section=section,
                            page=page,
                            line_start=None if page else text.count("\n", 0, start) + 1,
                            line_end=None if page else text.count("\n", 0, max(start, end - 1)) + 1,
                            start=start,
                            end=end,
                            source_url=source_url,
                        )
                    )
                if end == finish:
                    break
                start = max(start + 1, end - overlap)
    if not chunks:
        raise ValueError("文档没有可读取的文字。扫描 PDF 请先做 OCR，再上传。")
    return chunks
