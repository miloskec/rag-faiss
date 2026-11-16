import pymupdf
import pytest
from docx import Document
from odf.opendocument import OpenDocumentText
from odf.text import P

from doc_indexer import extract_sections_from_odt, extract_sections_from_word
from index_utils import chunk_sections
from pdf_indexer import extract_sections_from_pdf, is_heading


def test_pymupdf_heading_flag_uses_bold_bit():
    assert is_heading({"size": 10, "flags": 16})
    assert not is_heading({"size": 10, "flags": 1})
    assert is_heading({"size": 14, "flags": 0})


def test_pdf_extraction_keeps_heading_and_body(tmp_path):
    path = tmp_path / "guide.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Firmware requirement", fontsize=18)
    page.insert_text((72, 105), "Aster-100 requires firmware 2.4.0.", fontsize=10)
    document.save(path)
    document.close()

    sections = extract_sections_from_pdf(path)

    assert any("firmware 2.4.0" in section["text"] for section in sections)
    assert any(section["page"] == 1 for section in sections)


def test_chunking_preserves_source_identity_and_is_stable():
    sections = [{"title": "Firmware", "text": "word " * 100, "page": 2}]

    first = chunk_sections(sections, "device.md", chunk_size=80, overlap=10)
    second = chunk_sections(sections, "device.md", chunk_size=80, overlap=10)

    assert first == second
    assert len(first) > 1
    assert first[0]["source"] == "device.md"
    assert first[0]["chunk_id"] == "device.md::0000"
    assert first[1]["chunk_id"] == "device.md::0001"
    assert all(chunk["text"].split()[0] == "word" for chunk in first)


def test_word_heading_and_body_extraction(tmp_path):
    path = tmp_path / "guide.docx"
    document = Document()
    document.add_heading("Model identity", level=1)
    document.add_paragraph("The synthetic model is Aster-100.")
    document.save(path)

    sections = extract_sections_from_word(path)

    assert sections == [
        {"title": "Model identity", "text": "The synthetic model is Aster-100.", "page": None}
    ]


def test_odt_heading_and_body_extraction(tmp_path):
    path = tmp_path / "guide.odt"
    document = OpenDocumentText()
    document.text.addElement(
        P(stylename="Heading_20_1", text="Firmware requirement")
    )
    document.text.addElement(P(text="Firmware 2.4.0 is required."))
    document.save(str(path))

    sections = extract_sections_from_odt(path)

    assert sections == [
        {
            "title": "Firmware requirement",
            "text": "Firmware 2.4.0 is required.",
            "page": None,
        }
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_chunk_configuration_fails(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_sections(
            [{"title": "Heading", "text": "Some text"}],
            "source.md",
            chunk_size=chunk_size,
            overlap=overlap,
        )
