import faiss
import numpy as np
import pytest

from index_utils import (
    IndexCompatibilityError,
    load_index_state,
    save_index_state,
)


def test_index_and_json_mapping_round_trip(tmp_path):
    index = faiss.IndexFlatL2(3)
    index.add(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    mappings = [
        {"chunk_id": "a::0000", "source": "a.md", "text": "alpha"},
        {"chunk_id": "b::0000", "source": "b.md", "text": "beta"},
    ]
    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "text_mapping.json"

    save_index_state(index, mappings, index_path, mapping_path)
    loaded_index, loaded_mappings = load_index_state(
        index_path, mapping_path, dimension=3
    )

    assert loaded_index.ntotal == 2
    assert loaded_mappings == mappings


def test_missing_index_state_is_an_empty_index(tmp_path):
    index, mappings = load_index_state(
        tmp_path / "index.faiss",
        tmp_path / "text_mapping.json",
        dimension=3,
    )

    assert index.ntotal == 0
    assert mappings == []


def test_dimension_mismatch_fails_explicitly(tmp_path):
    index = faiss.IndexFlatL2(3)
    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "text_mapping.json"
    faiss.write_index(index, str(index_path))
    mapping_path.write_text("[]", encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match="dimension"):
        load_index_state(index_path, mapping_path, dimension=4)


def test_vector_mapping_count_mismatch_fails(tmp_path):
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "text_mapping.json"
    faiss.write_index(index, str(index_path))
    mapping_path.write_text("[]", encoding="utf-8")

    with pytest.raises(IndexCompatibilityError, match="mapping"):
        load_index_state(index_path, mapping_path, dimension=2)
