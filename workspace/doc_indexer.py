from pathlib import Path

import docx
import faiss
from odf import teletype
from odf.opendocument import load
from odf.text import P

import shared_state
from index_utils import (
    IndexCompatibilityError,
    chunk_sections,
    load_index_state,
    save_index_state,
)
from pdf_indexer import _encode_passages
from rag_config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_DIMENSION


DOC_INDEX_PATH = DATA_DIR / "doc" / "index.faiss"
DOC_MAPPING_PATH = DATA_DIR / "doc" / "text_mapping.json"


def _flush_section(sections, title, content):
    normalized = " ".join(content.split())
    if normalized:
        sections.append({"title": title or "Introduction", "text": normalized, "page": None})


def extract_sections_from_word(docx_path):
    sections = []
    current_heading = "Introduction"
    current_content = []
    for paragraph in docx.Document(docx_path).paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if is_heading_word(paragraph):
            _flush_section(sections, current_heading, " ".join(current_content))
            current_heading = text
            current_content = []
        else:
            current_content.append(text)
    _flush_section(sections, current_heading, " ".join(current_content))
    return sections


def extract_sections_from_odt(odt_path):
    sections = []
    current_heading = "Introduction"
    current_content = []
    document = load(str(odt_path))
    for paragraph in document.getElementsByType(P):
        text = teletype.extractText(paragraph).strip()
        if not text:
            continue
        if is_heading_odt(paragraph):
            _flush_section(sections, current_heading, " ".join(current_content))
            current_heading = text
            current_content = []
        else:
            current_content.append(text)
    _flush_section(sections, current_heading, " ".join(current_content))
    return sections


def is_heading_word(paragraph):
    style = paragraph.style.name if paragraph.style is not None else ""
    return style.startswith("Heading")


def is_heading_odt(paragraph):
    style = paragraph.getAttribute("stylename")
    if not style:
        style = paragraph.attributes.get("text:style-name", "")
    return str(style).startswith("Heading")


def extract_sections_for_path(path):
    if path.suffix.lower() == ".docx":
        return extract_sections_from_word(path)
    if path.suffix.lower() == ".odt":
        return extract_sections_from_odt(path)
    return []


def add_chunks(chunks, embedding_model=None, index=None, mappings=None):
    index = shared_state.index_doc if index is None else index
    mappings = shared_state.indexed_text_mapping_doc if mappings is None else mappings
    if not chunks:
        return 0
    vectors = _encode_passages(chunks, embedding_model)
    if vectors.shape[1] != index.d:
        raise IndexCompatibilityError(
            f"Embedding dimension is {vectors.shape[1]}, index expects {index.d}"
        )
    index.add(vectors)
    mappings.extend(chunks)
    return len(chunks)


def update_doc_index(doc_folder, embedding_model=None, index=None, mappings=None):
    chunks = []
    paths = sorted(
        [
            *Path(doc_folder).glob("*.docx"),
            *Path(doc_folder).glob("*.odt"),
        ],
        key=lambda item: item.name,
    )
    for path in paths:
        sections = extract_sections_for_path(path)
        chunks.extend(
            chunk_sections(
                sections,
                source=path,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        )
    return add_chunks(chunks, embedding_model, index, mappings)


def rebuild_doc_index(doc_folder, embedding_model=None):
    new_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    new_mappings = []
    count = update_doc_index(
        doc_folder,
        embedding_model=embedding_model,
        index=new_index,
        mappings=new_mappings,
    )
    save_index_state(new_index, new_mappings, DOC_INDEX_PATH, DOC_MAPPING_PATH)
    shared_state.index_doc = new_index
    shared_state.indexed_text_mapping_doc = new_mappings
    return count


def clear_index():
    shared_state.index_doc.reset()
    shared_state.indexed_text_mapping_doc.clear()


def load_index(
    index_path=DOC_INDEX_PATH,
    mapping_path=DOC_MAPPING_PATH,
    dimension=EMBEDDING_DIMENSION,
):
    return load_index_state(index_path, mapping_path, dimension)


def initialize():
    shared_state.index_doc, shared_state.indexed_text_mapping_doc = load_index()


def save_index_and_mappings():
    save_index_state(
        shared_state.index_doc,
        shared_state.indexed_text_mapping_doc,
        DOC_INDEX_PATH,
        DOC_MAPPING_PATH,
    )
