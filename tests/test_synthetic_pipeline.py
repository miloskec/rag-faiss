from pathlib import Path

import faiss

import pdf_indexer
import searcher
import shared_state
from fakes import CapturingModelHandler, HashEmbedder


def test_synthetic_corpus_indexes_and_retrieves_source_aligned_candidate():
    corpus = (
        Path(__file__).resolve().parents[1]
        / "workspace"
        / "examples"
        / "synthetic_corpus"
    )
    dimension = 64
    embedder = HashEmbedder(dimension)
    shared_state.index = faiss.IndexFlatL2(dimension)
    shared_state.indexed_text_mapping = []

    count = pdf_indexer.update_text_index(corpus, embedding_model=embedder)
    candidates = searcher.retrieve_candidates(
        "Which firmware version does Aster-100 require for calibration?",
        context_type="synthetic",
        top_k=5,
        embedder=embedder,
    )

    assert count == 9
    assert candidates
    assert candidates[0]["source"] == "aster-100.md"
    assert "2.4.0" in candidates[0]["context_text"]
    result = searcher.search_index(
        None,
        "Which firmware version does Aster-100 require for calibration?",
        context_type="synthetic",
        mode="rag_v0",
        embedder=embedder,
        model_handler=CapturingModelHandler(),
    )
    assert result["status"] == "ok"
    assert result["search_results"][0]["chunk_id"] == candidates[0]["chunk_id"]
