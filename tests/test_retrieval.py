import faiss
import numpy as np
import pytest

import model_utils
import searcher
import shared_state
from fakes import CapturingModelHandler, FakeEmbedder, FakeReranker


def _install_candidates():
    shared_state.index = faiss.IndexFlatL2(2)
    shared_state.index.add(
        np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    )
    shared_state.indexed_text_mapping = [
        {
            "title": "Aster-100",
            "text": "Requires firmware 2.4.0.",
            "source": "aster-100.md",
            "chunk_id": "aster-100.md::0001",
            "page": 1,
        },
        {
            "title": "Aster-200",
            "text": "Requires firmware 3.1.0.",
            "source": "aster-200.md",
            "chunk_id": "aster-200.md::0001",
            "page": 1,
        },
        {
            "title": "Calibration",
            "text": "Run the calibration self-test.",
            "source": "field-calibration.md",
            "chunk_id": "field-calibration.md::0001",
            "page": 1,
        },
    ]


def test_requesting_more_vectors_than_available_filters_faiss_minus_one():
    _install_candidates()
    embedder = FakeEmbedder([[1.0, 0.0]])

    results = searcher.retrieve_candidates(
        "Aster-100 firmware", top_k=10, embedder=embedder
    )

    assert len(results) == 3
    assert all(result["index"] >= 0 for result in results)
    assert results[0]["source"] == "aster-100.md"
    assert results[0]["vector_rank"] == 1


def test_empty_index_returns_no_candidates():
    shared_state.index = faiss.IndexFlatL2(2)
    shared_state.indexed_text_mapping = []

    result = searcher.retrieve_candidates(
        "anything", embedder=FakeEmbedder([[1.0, 0.0]])
    )

    assert result == []


def test_reranker_receives_explicit_query_passage_pairs_and_changes_order():
    _install_candidates()
    candidates = searcher.retrieve_candidates(
        "Aster-100 firmware",
        top_k=2,
        embedder=FakeEmbedder([[1.0, 0.0]]),
    )
    reranker = FakeReranker([0.2, 0.9])

    reranked = searcher.rerank_candidates(
        "Aster-100 firmware", candidates, reranker
    )

    assert reranker.pairs[0][0] == "Aster-100 firmware"
    assert "Aster-100" in reranker.pairs[0][1]
    assert reranked[0]["source"] == "aster-200.md"
    assert reranked[0]["rank"] == 1


def test_context_has_source_headers_and_respects_character_budget():
    _install_candidates()
    candidates = searcher.retrieve_candidates(
        "firmware", top_k=3, embedder=FakeEmbedder([[1.0, 0.0]])
    )

    context = searcher.assemble_context(candidates, max_chars=120)

    assert len(context) <= 120
    assert context.startswith("[Source: aster-100.md; chunk:")


def test_context_limit_includes_truncation_marker():
    results = [
        {
            "source": "long.md",
            "chunk_id": "long.md::0000",
            "context_text": "A" * 2000,
        }
    ]

    context = searcher.assemble_context(results, max_chars=90)

    assert len(context) <= 90
    assert context.endswith(" [...]")


def test_rag_v0_and_rag_v1_share_pipeline_but_use_different_order():
    _install_candidates()
    embedder = FakeEmbedder([[1.0, 0.0]])
    reranker = FakeReranker([0.1, 0.9, 0.0])
    handler = CapturingModelHandler()

    v0 = searcher.search_index(
        None,
        "Which firmware does Aster-100 require?",
        context_type="pdf",
        top_k=3,
        mode="rag_v0",
        embedder=embedder,
        reranker=reranker,
        model_handler=handler,
    )
    v1 = searcher.search_index(
        None,
        "Which firmware does Aster-100 require?",
        context_type="pdf",
        top_k=3,
        mode="rag_v1",
        embedder=embedder,
        reranker=reranker,
        model_handler=handler,
    )

    assert v0["search_results"][0]["source"] == "aster-100.md"
    assert v1["search_results"][0]["source"] == "aster-200.md"
    assert v0["retrieval_results"] == v1["retrieval_results"]
    assert "Question: Which firmware does Aster-100 require?" in handler.prompt
    assert "[Source:" in handler.prompt


def test_no_context_skips_generation():
    shared_state.index = faiss.IndexFlatL2(2)
    shared_state.indexed_text_mapping = []
    handler = CapturingModelHandler()

    result = searcher.search_index(
        None,
        "No matching context",
        embedder=FakeEmbedder([[1.0, 0.0]]),
        model_handler=handler,
    )

    assert result["status"] == "no_context"
    assert result["model_response"] is None
    assert handler.prompt is None


def test_invalid_model_selection_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported model"):
        model_utils.get_model_handler("does-not-exist")
