import numpy as np

import shared_state
from index_utils import IndexCompatibilityError, display_text
from model_utils import get_model_handler, query_model, unload_current_model
from rag_config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    MAX_CONTEXT_CHARS,
    RERANKER_MODEL_NAME,
    normalize_rag_mode,
)


_embedding_model = None
_reranker_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


def unload_model():
    return unload_current_model()


def encode_query(query, embedder=None):
    model = embedder or get_embedding_model()
    vector = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    return vector


def _state_for_context(context_type):
    normalized = (context_type or "pdf").strip().lower()
    if normalized in {"pdf", "synthetic"}:
        return shared_state.index, shared_state.indexed_text_mapping
    if normalized in {"doc", "docx", "odt"}:
        return shared_state.index_doc, shared_state.indexed_text_mapping_doc
    raise ValueError("context_type must be one of: pdf, synthetic, doc")


def _normalize_mapping(entry):
    if isinstance(entry, str):
        return {
            "text": entry,
            "title": "Legacy mapping",
            "source": "legacy",
            "chunk_id": "legacy",
            "page": None,
        }
    return dict(entry)


def retrieve_candidates(
    query,
    context_type="pdf",
    top_k=DEFAULT_TOP_K,
    embedder=None,
):
    query = (query or "").strip()
    if not query:
        raise ValueError("A non-empty query is required")
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise ValueError("top_k must be an integer between 1 and 100")

    index, mappings = _state_for_context(context_type)
    if index.ntotal == 0 or not mappings:
        return []
    if index.ntotal != len(mappings):
        raise IndexCompatibilityError(
            f"FAISS has {index.ntotal} vectors but mapping has {len(mappings)} entries"
        )

    query_vector = encode_query(query, embedder)
    if query_vector.shape[1] != index.d:
        raise IndexCompatibilityError(
            f"Query embedding dimension is {query_vector.shape[1]}, index expects {index.d}"
        )

    distances, indices = index.search(query_vector, top_k)
    candidates = []
    for vector_rank, (distance, mapped_index) in enumerate(
        zip(distances[0], indices[0]), start=1
    ):
        if mapped_index < 0 or mapped_index >= len(mappings):
            continue
        mapping = _normalize_mapping(mappings[int(mapped_index)])
        mapping["context_text"] = display_text(mapping)
        mapping["index"] = int(mapped_index)
        mapping["vector_rank"] = vector_rank
        mapping["distance"] = float(distance)
        mapping["retrieval_score"] = float(distance)
        candidates.append(mapping)
    return candidates


def rerank_candidates(query, candidates, reranker=None):
    if not candidates:
        return []
    model = reranker or get_reranker()
    # Cross-encoders consume explicit (query, passage) pairs.
    pairs = [[query, candidate["context_text"]] for candidate in candidates]
    scores = np.asarray(model.predict(pairs), dtype=float).reshape(-1)
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned a different number of scores than candidates")

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        reranked.append(item)
    reranked.sort(
        key=lambda item: (item["rerank_score"], -item["vector_rank"]),
        reverse=True,
    )
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
    return reranked


def assemble_context(results, max_chars=MAX_CONTEXT_CHARS):
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    pieces = []
    used = 0
    for result in results:
        header = (
            f"[Source: {result.get('source', 'unknown')}; "
            f"chunk: {result.get('chunk_id', 'unknown')}]"
        )
        body = result.get("context_text") or display_text(result)
        candidate = f"{header}\n{body}"
        separator = "\n\n" if pieces else ""
        remaining = max_chars - used - len(separator)
        if remaining <= len(header) + 1:
            break
        if len(candidate) > remaining:
            available_body = remaining - len(header) - 1
            candidate = f"{header}\n{body[:available_body].rstrip()} [...]"
        pieces.append(candidate)
        used += len(separator) + len(candidate)
        if used >= max_chars:
            break
    return "\n\n".join(pieces)


def search_index(
    aimodel,
    query,
    template=None,
    context_type="pdf",
    top_k=DEFAULT_TOP_K,
    mode=None,
    embedder=None,
    reranker=None,
    model_handler=None,
    max_context_chars=MAX_CONTEXT_CHARS,
):
    mode = normalize_rag_mode(mode)
    candidates = retrieve_candidates(query, context_type, top_k, embedder)
    if not candidates:
        return {
            "mode": mode,
            "status": "no_context",
            "context": "",
            "retrieval_results": [],
            "search_results": [],
            "model_response": None,
        }

    if mode == "rag_v1":
        ordered_results = rerank_candidates(query, candidates, reranker)
    else:
        ordered_results = []
        for rank, candidate in enumerate(candidates, start=1):
            item = dict(candidate)
            item["rank"] = rank
            ordered_results.append(item)

    context = assemble_context(ordered_results, max_context_chars)
    handler = model_handler or get_model_handler(aimodel)
    response = query_model(handler, query, context, template)
    return {
        "mode": mode,
        "status": "ok",
        "context": context,
        "retrieval_results": candidates,
        "search_results": ordered_results,
        "model_response": response,
    }
