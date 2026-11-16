import json
import os
from pathlib import Path

import faiss


class IndexCompatibilityError(RuntimeError):
    pass


def split_text(text, chunk_size=1200, overlap=150):
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks = []
    start = 0
    while start < len(normalized):
        hard_end = min(start + chunk_size, len(normalized))
        end = hard_end
        if hard_end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, hard_end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        next_start = max(end - overlap, start + 1)
        word_boundary = normalized.find(" ", next_start, end)
        if word_boundary != -1:
            next_start = word_boundary + 1
        start = next_start
    return chunks


def chunk_sections(
    sections,
    source,
    chunk_size=1200,
    overlap=150,
):
    source_name = Path(source).name
    chunks = []
    ordinal = 0
    for section in sections:
        title = (section.get("title") or "Untitled").strip()
        for piece in split_text(section.get("text", ""), chunk_size, overlap):
            chunks.append(
                {
                    "source": source_name,
                    "chunk_id": f"{source_name}::{ordinal:04d}",
                    "title": title,
                    "text": piece,
                    "page": section.get("page"),
                }
            )
            ordinal += 1
    return chunks


def embedding_text(chunk):
    return f"{chunk['title']}\n{chunk['text']}".strip()


def display_text(chunk):
    title = chunk.get("title", "").strip()
    text = chunk.get("text", "").strip()
    if title and not text.startswith(title):
        return f"{title}: {text}"
    return text or title


def _temporary_path(path):
    path = Path(path)
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def save_index_state(index, mappings, index_path, mapping_path):
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    if index.ntotal != len(mappings):
        raise IndexCompatibilityError(
            f"FAISS has {index.ntotal} vectors but mapping has {len(mappings)} entries"
        )

    temporary_index = _temporary_path(index_path)
    temporary_mapping = _temporary_path(mapping_path)
    try:
        faiss.write_index(index, str(temporary_index))
        with temporary_mapping.open("w", encoding="utf-8") as handle:
            json.dump(mappings, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_index, index_path)
        os.replace(temporary_mapping, mapping_path)
    finally:
        for temporary in (temporary_index, temporary_mapping):
            if temporary.exists():
                temporary.unlink()


def load_index_state(index_path, mapping_path, dimension):
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)
    if not index_path.exists() and not mapping_path.exists():
        return faiss.IndexFlatL2(dimension), []
    if not index_path.exists() or not mapping_path.exists():
        raise IndexCompatibilityError(
            f"Index and mapping must exist together: {index_path}, {mapping_path}"
        )

    index = faiss.read_index(str(index_path))
    if index.d != dimension:
        raise IndexCompatibilityError(
            f"FAISS index dimension is {index.d}, expected {dimension}; rebuild the index"
        )
    with mapping_path.open("r", encoding="utf-8") as handle:
        mappings = json.load(handle)
    if index.ntotal != len(mappings):
        raise IndexCompatibilityError(
            f"FAISS has {index.ntotal} vectors but mapping has {len(mappings)} entries"
        )
    return index, mappings
