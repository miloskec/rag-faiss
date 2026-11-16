import faiss

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
    monkeypatch.setattr(app_module.pdf_indexer, "clear_index", lambda: None)
    monkeypatch.setattr(
        app_module.pdf_indexer, "update_text_index", lambda folder: 3
    )
    monkeypatch.setattr(app_module.pdf_indexer, "save_index_and_mappings", lambda: None)
    client = app_module.app.test_client()

    get_response = client.get("/update-index?type=synthetic")
    post_response = client.post("/update-index?type=synthetic")

    assert get_response.status_code == 405
    assert post_response.status_code == 200
    assert post_response.get_json()["chunks_indexed"] == 3


def test_synthetic_api_flow_uses_retrieval_and_generation_double(monkeypatch):
    embedder = HashEmbedder(64)
    shared_state.index = faiss.IndexFlatL2(64)
    shared_state.indexed_text_mapping = []
    monkeypatch.setattr(app_module.pdf_indexer, "get_embedding_model", lambda: embedder)
    monkeypatch.setattr(app_module.searcher, "get_embedding_model", lambda: embedder)
    handler = CapturingModelHandler()
    monkeypatch.setattr(app_module.searcher, "get_model_handler", lambda model: handler)
    monkeypatch.setattr(app_module.pdf_indexer, "save_index_and_mappings", lambda: None)
    client = app_module.app.test_client()

    update = client.post("/update-index?type=synthetic")
    search = client.post(
        "/search",
        json={
            "query": "Which firmware version does Aster-100 require?",
            "context_type": "synthetic",
            "mode": "v0",
            "model": "fake-model",
        },
    )

    assert update.status_code == 200
    assert update.get_json()["chunks_indexed"] >= 6
    assert search.status_code == 200
    payload = search.get_json()
    assert payload["search_results"][0]["source"] == "aster-100.md"
    assert payload["model_response"] == "deterministic test response"


def test_unwritable_log_directory_falls_back_without_crashing(tmp_path, monkeypatch):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(app_module, "LOG_DIR", blocker)

    app_module.configure_logging()
