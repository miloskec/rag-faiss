from pathlib import Path

import faiss
import numpy as np
import pymupdf

import shared_state
from index_utils import (
    IndexCompatibilityError,
    chunk_sections,
    embedding_text,
    load_index_state,
    save_index_state,
)
from rag_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
)


PDF_INDEX_PATH = DATA_DIR / "pdf" / "index.faiss"
PDF_MAPPING_PATH = DATA_DIR / "pdf" / "text_mapping.json"

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def is_heading(span, minimum_size=14):
    # PyMuPDF flag bit 16 marks bold text; bit 1 marks superscript.
    flags = int(span.get("flags", 0))
    return float(span.get("size", 0)) >= minimum_size or bool(flags & 16)


def _flush_section(sections, title, content, page):
    normalized = " ".join(content.split())
    if normalized:
        sections.append({"title": title or "Introduction", "text": normalized, "page": page})


def extract_sections_from_pdf(pdf_path):
    sections = []
    current_heading = "Introduction"
    current_content = ""
    current_page = 1

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = " ".join(span.get("text", "").strip() for span in spans).strip()
                    if not text:
                        continue
                    if any(is_heading(span) for span in spans):
                        _flush_section(
                            sections, current_heading, current_content, current_page
                        )
                        current_heading = text
                        current_content = ""
                        current_page = page_number
                    else:
                        current_content += f" {text}"

    _flush_section(sections, current_heading, current_content, current_page)
    return sections


def extract_sections_from_text(text_path):
    sections = []
    current_heading = "Introduction"
    current_content = []

    with Path(text_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                current_content.append("")
                continue
            if stripped.startswith("#"):
                _flush_section(
                    sections, current_heading, " ".join(current_content), 1
                )
                current_heading = stripped.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(stripped)

    _flush_section(sections, current_heading, " ".join(current_content), 1)
    return sections


def _encode_passages(chunks, embedding_model=None):
    model = embedding_model or get_embedding_model()
    passages = [f"passage: {embedding_text(chunk)}" for chunk in chunks]
    vectors = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors


def add_chunks(chunks, embedding_model=None, index=None, mappings=None):
    index = shared_state.index if index is None else index
    mappings = shared_state.indexed_text_mapping if mappings is None else mappings
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


def update_pdf_index(pdf_folder, embedding_model=None, index=None, mappings=None):
    chunks = []
    for path in sorted(Path(pdf_folder).glob("*.pdf"), key=lambda item: item.name):
        sections = extract_sections_from_pdf(path)
        chunks.extend(
            chunk_sections(
                sections,
                source=path,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        )
    return add_chunks(chunks, embedding_model, index, mappings)


def update_text_index(text_folder, embedding_model=None, index=None, mappings=None):
    chunks = []
    paths = sorted(
        [*Path(text_folder).glob("*.md"), *Path(text_folder).glob("*.txt")],
        key=lambda item: item.name,
    )
    for path in paths:
        sections = extract_sections_from_text(path)
        chunks.extend(
            chunk_sections(
                sections,
                source=path,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        )
    return add_chunks(chunks, embedding_model, index, mappings)


def rebuild_pdf_index(pdf_folder, embedding_model=None):
    new_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    new_mappings = []
    count = update_pdf_index(
        pdf_folder,
        embedding_model=embedding_model,
        index=new_index,
        mappings=new_mappings,
    )
    save_index_state(new_index, new_mappings, PDF_INDEX_PATH, PDF_MAPPING_PATH)
    shared_state.index = new_index
    shared_state.indexed_text_mapping = new_mappings
    return count


def rebuild_text_index(text_folder, embedding_model=None):
    new_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    new_mappings = []
    count = update_text_index(
        text_folder,
        embedding_model=embedding_model,
        index=new_index,
        mappings=new_mappings,
    )
    save_index_state(new_index, new_mappings, PDF_INDEX_PATH, PDF_MAPPING_PATH)
    shared_state.index = new_index
    shared_state.indexed_text_mapping = new_mappings
    return count


def clear_index():
    shared_state.index.reset()
    shared_state.indexed_text_mapping.clear()


def load_index(
    index_path=PDF_INDEX_PATH,
    mapping_path=PDF_MAPPING_PATH,
    dimension=EMBEDDING_DIMENSION,
):
    return load_index_state(index_path, mapping_path, dimension)


def initialize():
    shared_state.index, shared_state.indexed_text_mapping = load_index()


def save_index_and_mappings():
    save_index_state(
        shared_state.index,
        shared_state.indexed_text_mapping,
        PDF_INDEX_PATH,
        PDF_MAPPING_PATH,
    )
