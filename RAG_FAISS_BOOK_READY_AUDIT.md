# RAG-FAISS Book-Ready Audit

Audit date: 2026-11-16

Repository: `miloskec/rag-faiss`

Working branch: `book-ready/rag-v0-v1`

Default branch and historical commits were preserved.

## Executive verdict

**Status: COMPLETE — BOOK-READY WITH DOCUMENTED LIMITATIONS.**

The repository now provides a technically defensible introductory dense-vector
RAG pipeline with two explicitly separate modes:

1. **RAG v0:** FAISS candidates go directly to context assembly.
2. **RAG v1:** the same candidates are passed through a cross-encoder before
   context assembly.

The historical architecture remains recognizable. The source now has
deterministic tests for ingestion, chunk/source identity, index persistence,
invalid FAISS results, reranker input, mode separation, prompt construction,
context limits, model validation, and Flask input handling.

The book can use this as a basic RAG starting point. It must not describe the
implementation as production-ready or as Universal Hybrid RAG.

## Actual architecture

```text
PDF / DOCX / ODT / Markdown
  -> heading and section extraction
  -> deterministic overlapping chunks
  -> E5 passage embeddings
  -> normalized vectors in FAISS IndexFlatL2
  -> top-K candidate retrieval
  -> RAG v0: vector order
     RAG v1: cross-encoder ordering of the same candidate pool
  -> source-labelled context
  -> model-specific grounded prompt
  -> local causal language model
  -> response plus retrieval metadata
```

The principal runtime modules are:

- `workspace/pdf_indexer.py`: PDF/Markdown extraction, chunking, embeddings,
  the dense index, and persistence.
- `workspace/doc_indexer.py`: DOCX/ODT extraction and the office-document
  index.
- `workspace/index_utils.py`: chunking, mapping schema, atomic persistence,
  and compatibility validation.
- `workspace/searcher.py`: candidate retrieval, optional reranking, context
  assembly, and mode orchestration.
- `workspace/prompting.py`: grounding instructions and fallback prompt styles.
- `workspace/model_utils.py`: model selection, cache, unloading, and prompt
  dispatch.
- `workspace/base_model_handler.py`: local/remote model loading and inference.
- `workspace/app.py`: local Flask API and error mapping.

## Audit method

The audit inspected the actual current branch, source, Git history, generated
artifacts, configuration, Docker build definition, and available tests.
Findings were reproduced in code and, where possible, converted into focused
regression tests before remediation.

No large language model, E5 embedding model, or cross-encoder weights were
downloaded. Heavy inference interfaces were replaced with deterministic test
doubles. This limitation is intentional and is separated from claims that were
actually verified.

## Confirmed findings

### P0 — Correctness and runtime integrity

| ID | Finding | Evidence in the audited version | Resolution |
| --- | --- | --- | --- |
| P0-1 | PDF bold detection used the superscript bit | `is_heading` tested `span["flags"] & 1`; PyMuPDF uses bit 16 for bold | Corrected to `flags & 16`; size threshold remains a documented heuristic |
| P0-2 | FAISS `-1` indices were not filtered | `searcher.py` indexed mappings with every returned FAISS ID | Invalid and out-of-range indices are skipped |
| P0-3 | The configured “reranker” was an embedding model used as `text-classification` | `intfloat/multilingual-e5-large` was passed to a classification pipeline | Replaced with `sentence_transformers.CrossEncoder` and explicit `(query, passage)` pairs |
| P0-4 | Candidate order was not exposed as an educational mode | Every search always reranked and returned only initial result records | Added `rag_v0` and `rag_v1`; both modes return initial retrieval records and final ordering |
| P0-5 | Prompt question could be lost by tokenizer truncation | Context preceded the question and the whole prompt could be truncated | Grounding instruction and question precede context; model chat templates are preferred |
| P0-6 | Retrieval mappings had no reliable identity | Mappings were concatenated strings containing a filename suffix | JSON mappings now include `source`, `chunk_id`, `title`, `text`, and page when available |
| P0-7 | Invalid model selection could reach inference as a dictionary | `get_model_handler` returned `{"error": ...}` | Unknown models now raise `ValueError`; API returns an explicit 400 |
| P0-8 | The DOC index used 768 dimensions while query embeddings used 1024 | Historical DOC index header was 768; shared state and E5 were 1024 | Historical artifacts were separated; runtime checks dimension and vector/mapping counts explicitly |
| P0-9 | Heavy embedding and reranker models loaded at module import | `SentenceTransformer` and `pipeline` were constructed during import | Embedding and reranker models are initialized lazily |
| P0-10 | Local model detection rejected valid Llama-style tokenizers | It required `tokenizer.json`, excluding models that use `tokenizer.model` | Detection now accepts either tokenizer format and requires model weights |

### P1 — Reproducibility, safety, and reader usability

| ID | Finding | Evidence in the audited version | Resolution |
| --- | --- | --- | --- |
| P1-1 | Docker installed an unpinned dependency set on every start | Startup ran a long `pip3 install` command with floating versions | Added pinned `requirements.txt`, `.dockerignore`, a pinned base-image digest, and image-build installation |
| P1-2 | Flask debug and mutating GET endpoints were unsafe defaults | `debug=True`; index update and cleanup used GET | Debug defaults to false; mutating endpoints are POST; local bind defaults to `127.0.0.1` |
| P1-3 | Index persistence allowed partial or mismatched state | Pickle mapping, no dimension check, no mapping-count check | Each pair is written into a complete generation directory; one atomically replaced manifest publishes the active pair |
| P1-4 | Relative paths depended on container working directory | Paths used `./workspace/...` | Runtime paths are anchored to the module directory |
| P1-5 | E5 retrieval prefixes were absent | Embeddings encoded raw text | Passages use `passage:` and queries use `query:`; vectors are normalized |
| P1-6 | Qwen and SambaLingo choices were base models | The prior commit selected `Qwen2.5-7B` and `SambaLingo-Serbian-Base` | Defaults now use `Qwen2.5-7B-Instruct` and `SambaLingo-Serbian-Chat` |
| P1-7 | README differed materially from runtime behavior | It still described PCI-DSS content and old model/API behavior | README now documents the synthetic corpus, modes, configuration, models, API, Docker, and limitations |
| P1-8 | Tests were absent | No test suite or test dependency existed | Added focused pytest coverage and deterministic fakes |
| P1-9 | API input errors were inconsistently validated | Non-object JSON, non-string context types, invalid top-K, and malformed templates could reach deeper code | Request types and ranges are validated before retrieval/model work |
| P1-10 | `.env` was loaded after modules had already read configuration | `rag_config` constants could ignore values in `.env` | The configuration module now loads the project `.env` before reading variables |
| P1-11 | An unwritable log directory prevented startup | A local launch failed before Flask started when `workspace/logs` was not writable | File logging falls back to stderr and the failure is logged |
| P1-12 | Context truncation could exceed the character budget | The ` [...]` marker was appended after calculating the body allowance | Marker length is included in the remaining budget and covered by a long-passage regression test |
| P1-13 | Formatted chat templates could receive duplicate special tokens | A `tokenize=False` template was tokenized again with default special-token insertion | Prepared prompts carry an explicit tokenizer policy; chat-template prompts use `add_special_tokens=False` |
| P1-14 | A failed index rebuild cleared the active in-memory index | `/update-index` cleared shared state before ingestion completed | Rebuilds use an isolated FAISS/mapping pair and replace shared state only after persistence succeeds |
| P1-15 | Historical feed documents were included in the Docker build | `COPY workspace` included the Ustav PDF despite unresolved rights | `workspace/feed/` and archived artifacts are excluded from Docker build context |
| P1-16 | Hugging Face authentication names were inconsistent | Compose translated a non-canonical name while local Python did not pass it explicitly | The project now uses only `HF_TOKEN`, loads `.env` before Transformers, and passes the token to model loading |
| P1-17 | Relative log paths depended on the process working directory | `workspace/logs` resolved from wherever the process started | Relative log paths resolve from the project root |
| P1-18 | The default reader path required gated Llama access | The default model was `meta-llama/Llama-2-7b-chat-hf` | The default is now public Apache-2.0 `mistral-7b-instruct`; Llama remains optional |

### Partial or unresolved

| ID | Finding | Status | Notes |
| --- | --- | --- | --- |
| R-1 | Historical fixtures may not be suitable for public distribution | Unresolved legal/ownership issue | Historical PDF and archived indexes remain in history. Their provenance and redistribution terms are not established by repository metadata. |
| R-2 | Third-party binary index artifacts are unsuitable runtime examples | Partially resolved | Old indexes and pickle mappings were moved to `workspace/legacy_artifacts/`, not deleted. New indexes are generated from synthetic text and ignored by Git. |
| R-3 | Pickle loading was unsafe for untrusted mappings | Resolved in runtime; historical file remains | Runtime uses JSON mappings. The historical pickle is archived evidence and is not loaded. |
| R-4 | Context can still lose lower-ranked evidence | Accepted introductory limitation | Character budget and tokenizer truncation are bounded, but the system does not estimate answer evidence coverage. |

## Rejected or non-material hypotheses

| Hypothesis | Classification | Reason |
| --- | --- | --- |
| The cross-encoder can improve corpus-wide recall | Rejected as an implementation claim | Reranking only reorders the existing top-K pool. It cannot retrieve an omitted passage. |
| Every retrieval failure is caused by reranking | Rejected | Chunking, dense recall, source ambiguity, prompt construction, and generation are independent failure sources. |
| The historical project must be replaced by an advanced hybrid architecture | Rejected as out of scope | The task requires a corrected introductory example, not an architectural rewrite. |
| The tracked Ustav PDF is automatically redistributable because the underlying law is public | Unverifiable | Public legal text does not by itself establish rights in a particular compiled/watermarked edition. |
| A synthetic example requires committed generated indexes | Rejected | Tests and documented indexing provide reproducibility while avoiding binary runtime artifacts. |

## Changes implemented

### Ingestion and chunking

- Corrected PyMuPDF bold-bit detection.
- Added one-line heading assembly instead of treating each span independently.
- Added deterministic overlapping character chunks with stable per-source order.
- Added structured page/source/chunk metadata.
- Added Markdown/plain-text ingestion for the synthetic corpus.
- Added DOCX and ODT heading extraction tests.

### Retrieval and reranking

- Kept FAISS `IndexFlatL2`.
- Normalized E5 vectors and added required prefixes.
- Filtered FAISS `-1` and out-of-range indices.
- Returned vector rank, distance, source, chunk ID, title, and text.
- Added `CrossEncoder.predict([[query, passage], ...])`.
- Added explicit `rag_v0` and `rag_v1` paths sharing retrieval and generation.
- Returned initial `retrieval_results` separately from final `search_results`.

### Context, prompting, and generation

- Added a character budget and source-labelled context.
- Placed the question before context in prompt text.
- Added tokenizer prompt limits and preserved the question under truncation.
- Preferred tokenizer-provided chat templates.
- Added Llama 2, Mistral, ChatML, SambaLingo, and plain fallback formats.
- Added explicit model error propagation instead of dictionary errors.
- Added valid local-model detection for both `tokenizer.json` and
  `tokenizer.model`.

### API, persistence, and packaging

- Converted mutating endpoints to POST.
- Added `/health`.
- Added request validation and explicit 400/409/500 responses.
- Replaced runtime pickle loading with JSON mapping loading.
- Added generation-based index publication behind one atomic manifest pointer.
- Added isolated index rebuilds so failed ingestion preserves the active index.
- Added index dimension and mapping-count validation.
- Added pinned requirements, Dockerfile, `.dockerignore`, and Compose health
  checks.
- Updated the Postman collection.

## Verification evidence

The following checks were executed in the repository's isolated `.venv`:

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **54 passed** |
| Python compile check with isolated bytecode directory | **passed** |
| Ruff static check for `workspace/` and `tests/` | **passed** |
| `git diff --check` | **passed** |
| Postman collection JSON parse | **passed** |
| `docker compose config` | **passed** |
| `docker build --check .` | **passed; no warnings** |
| `docker compose build` | **passed**; image `rag-faiss:book-ready` |
| Container dependency imports | **passed** |
| Container `python3 -m pip check` | **No broken requirements found** |
| Container pytest suite | **54 passed** |
| `docker compose` startup and `/health` request | **200 OK**; debug off |
| Local Flask startup and `/health` request | **200 OK**; logging fallback tested |
| Synthetic resolver | **9 chunks from 3 Markdown files** |
| Deterministic synthetic retrieval | Aster-100 firmware passage ranked first |
| Index persistence round trip | FAISS vectors and JSON mapping reloaded exactly |
| Oversized top-K against undersized index | invalid IDs filtered; only valid candidates returned |
| RAG v0/v1 comparison | identical initial candidate pool, different final order |
| Flask synthetic update/search integration | deterministic doubles executed the API flow |
| Long-passage context limit | truncation marker included; output stayed within budget |
| Formatted chat-template tokenization | `add_special_tokens=False` propagated through generation |
| Failed generation publication | previous manifest and active FAISS/mapping pair remained loadable |
| Failed index rebuild | active in-memory index remained unchanged |
| Docker build context | historical feed and archived artifacts excluded |

The tests do not claim:

- that E5 finds the correct passage for arbitrary real documents;
- that MiniLM improves real-world answer quality;
- that a selected language model produces factually correct prose;
- benchmark scores, latency, or recall improvements.

## Remaining risks and limitations

1. **No live heavyweight-model validation.** E5, the cross-encoder, and causal
   language models were not downloaded. Their library interfaces were exercised
   through deterministic doubles and source-level compatibility review.
2. **Dense-only recall.** Lexical exact matches, sparse representations,
   multi-vector retrieval, metadata filters, and query expansion are absent.
3. **Heuristic document parsing.** Complex PDF layouts, scanned PDFs, tables,
   and non-style-based headings may produce imperfect chunks.
4. **Evidence is traceable, not verified.** Source headers identify passages,
   but no claim-level entailment or citation verifier exists.
   The default MS MARCO cross-encoder is English, so Serbian reranking quality
   is unverified.
5. **Prompt injection remains possible.** Grounding instructions are not a
   security boundary.
6. **Local service security is minimal.** There is no authentication,
   authorization, TLS, rate limiting, or production WSGI deployment. The
   documented bind is localhost.
7. **Legal status is incomplete.** The README historically declares MIT, but
   no `LICENSE` file exists. The owner must resolve code and document licensing.
8. **Old index generations are retained.** Publication is consistent and the
   prior generation remains recoverable, but automatic retention/pruning
   policy is intentionally left to the operator.

## Final book-readiness decision

The code is suitable as an introductory RAG teaching example after the
implemented corrections. It is not suitable as a production reference
implementation, a retrieval-quality benchmark, or a Universal Hybrid RAG
example. The book should describe it as **basic dense-vector RAG with optional
cross-encoder reranking** and should explicitly teach the distinction between
candidate recall and candidate ordering.

## Verification appendix

Final branch, commit, test, build, and working-tree evidence is recorded in the
completion report after the final Git commit. Any post-commit verification must
be added here without changing the historical result above.
