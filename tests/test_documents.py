import pytest
from ai_docs_agent.documents import parse_document
from pypdf import PdfWriter


def test_markdown_sections_offsets_and_fenced_headings(tmp_path):
    path = tmp_path / "study.md"
    text = (
        "## 状态\n" + "状态用于恢复运行。" * 15 + "\n```python\n# example\n```\n## 工具\n工具可以调用 API。\n"
    )
    path.write_text(text, encoding="utf-8")
    chunks = parse_document(path, 80, 15)
    assert {chunk.section for chunk in chunks} == {"状态", "工具"}
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.line_start == text.count("\n", 0, chunk.start) + 1
        assert chunk.source == "study.md"
    assert any(
        chunks[i].end > chunks[i + 1].start
        for i in range(len(chunks) - 1)
        if chunks[i].section == chunks[i + 1].section
    )


def test_blank_scanned_pdf_has_actionable_error(tmp_path):
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(ValueError, match="OCR"):
        parse_document(path)


def test_invalid_parameters_and_encoding(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"\xff\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        parse_document(path)
    with pytest.raises(ValueError, match="参数"):
        parse_document(path, 50, 50)
