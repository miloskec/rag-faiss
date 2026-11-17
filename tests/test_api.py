import faiss
import numpy as np

import app as app_module
import shared_state
from fakes import CapturingModelHandler, HashEmbedder


def test_search_rejects_non_json_body():
    client = app_module.app.test_client()

    response = client.post("/search", data="not-json", content_type="text/plain")

    assert response.status_code == 400


def test_search_rejects_empty_query():
    client = app_module.app.test_client()

    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 400


def test_search_rejects_invalid_mode():
    client = app_module.app.test_client()

    response = client.post("/search", json={"query": "test", "mode": "v9"})

    assert response.status_code == 400


def test_search_rejects_legacy_model_even_without_context():
    client = app_module.app.test_client()

    response = client.post(
        "/search",
        json={"query": "test", "model": "mistral-7b-instruct"},
    )

    assert response.status_code == 400
    assert "Unsupported model" in response.get_json()["error"]


def test_search_returns_pipeline_result(monkeypatch):
    expected = {
        "mode": "rag_v0",
        "status": "ok",
        "context": "source context",
        "retrieval_results": [],
        "search_results": [],
        "model_response": "answer",
    }
    monkeypatch.setattr(app_module.searcher, "search_index", lambda *args, **kwargs: expected)
    client = app_module.app.test_client()

    response = client.post(
        "/search",
        json={"query": "test", "context_type": "synthetic", "mode": "v0"},
    )

    assert response.status_code == 200
    assert response.get_json()["model_response"] == "answer"


def test_update_index_is_post_only_and_returns_count(monkeypatch):
    monkeypatch.setattr(
        app_module.pdf_indexer, "rebuild_text_index", lambda folder: 3
    )
    client = app_module.app.test_client()

    get_response = client.get("/update-index?type=synthetic")
    post_response = client.post("/update-index?type=synthetic")

    assert get_response.status_code == 405
    assert post_response.status_code == 200
    assert post_response.get_json()["chunks_indexed"] == 3


def test_synthetic_api_flow_uses_retrieval_and_generation_double(
    tmp_path, monkeypatch
):
    embedder = HashEmbedder(64)
    shared_state.index = faiss.IndexFlatL2(64)
    shared_state.indexed_text_mapping = []
    monkeypatch.setattr(app_module.pdf_indexer, "EMBEDDING_DIMENSION", 64)
    monkeypatch.setattr(app_module.pdf_indexer, "PDF_INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(
        app_module.pdf_indexer, "PDF_MAPPING_PATH", tmp_path / "text_mapping.json"
    )
    monkeypatch.setattr(app_module.pdf_indexer, "get_embedding_model", lambda: embedder)
    monkeypatch.setattr(app_module.searcher, "get_embedding_model", lambda: embedder)
    handler = CapturingModelHandler()
    monkeypatch.setattr(app_module.searcher, "get_model_handler", lambda model: handler)
    client = app_module.app.test_client()

    update = client.post("/update-index?type=synthetic")
    search = client.post(
        "/search",
        json={
            "query": "Which firmware version does Aster-100 require?",
            "context_type": "synthetic",
            "mode": "v0",
        },
    )

    assert update.status_code == 200
    assert update.get_json()["chunks_indexed"] >= 6
    assert (tmp_path / "active_generation.json").exists()
    assert search.status_code == 200
    payload = search.get_json()
    assert payload["search_results"][0]["source"] == "aster-100.md"
    assert payload["model_response"] == "deterministic test response"


def test_unwritable_log_directory_falls_back_without_crashing(tmp_path, monkeypatch):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(app_module, "LOG_DIR", blocker)

    app_module.configure_logging()


def test_failed_rebuild_keeps_active_index(monkeypatch):
    active_index = faiss.IndexFlatL2(2)
    active_index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    shared_state.index = active_index
    shared_state.indexed_text_mapping = [
        {"chunk_id": "active::0000", "source": "active.md", "text": "active"}
    ]

    def fail_rebuild(folder):
        raise RuntimeError("simulated rebuild failure")

    monkeypatch.setattr(app_module.pdf_indexer, "rebuild_text_index", fail_rebuild)
    client = app_module.app.test_client()

    response = client.post("/update-index?type=synthetic")

    assert response.status_code == 500
    assert shared_state.index is active_index
    assert shared_state.indexed_text_mapping[0]["chunk_id"] == "active::0000"
