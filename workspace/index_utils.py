import json
import os
import uuid
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


def _sync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(root, generation):
    manifest_path = root / "active_generation.json"
    temporary_manifest = _temporary_path(manifest_path)
    try:
        _write_json(temporary_manifest, {"generation": generation})
        os.replace(temporary_manifest, manifest_path)
        _sync_directory(root)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()


def save_index_state(index, mappings, index_path, mapping_path):
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)
    if index_path.parent != mapping_path.parent:
        raise ValueError("Index and mapping must use the same parent directory")
    if index.ntotal != len(mappings):
        raise IndexCompatibilityError(
            f"FAISS has {index.ntotal} vectors but mapping has {len(mappings)} entries"
        )

    root = index_path.parent
    generation = uuid.uuid4().hex
    generation_dir = root / "generations" / generation
    generation_dir.mkdir(parents=True, exist_ok=False)
    generation_index = generation_dir / index_path.name
    generation_mapping = generation_dir / mapping_path.name
    temporary_index = _temporary_path(generation_index)
    temporary_mapping = _temporary_path(generation_mapping)
    try:
        faiss.write_index(index, str(temporary_index))
        _write_json(temporary_mapping, mappings)
        os.replace(temporary_index, generation_index)
        os.replace(temporary_mapping, generation_mapping)
        _sync_file(generation_index)
        _sync_file(generation_mapping)
        _write_manifest(root, generation)
    finally:
        for temporary in (temporary_index, temporary_mapping):
            if temporary.exists():
                temporary.unlink()


def _load_pair(index_path, mapping_path, dimension):
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


def load_index_state(index_path, mapping_path, dimension):
    index_path = Path(index_path)
    mapping_path = Path(mapping_path)
    if index_path.parent != mapping_path.parent:
        raise ValueError("Index and mapping must use the same parent directory")

    root = index_path.parent
    manifest_path = root / "active_generation.json"
    if not manifest_path.exists():
        return _load_pair(index_path, mapping_path, dimension)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    generation = manifest.get("generation")
    if not isinstance(generation, str) or Path(generation).name != generation:
        raise IndexCompatibilityError("Active index generation metadata is invalid")
    generation_dir = root / "generations" / generation
    return _load_pair(
        generation_dir / index_path.name,
        generation_dir / mapping_path.name,
        dimension,
    )
