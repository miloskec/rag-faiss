# RAG-FAISS Book Integration Guide

This document describes how the corrected repository supports the first two
book chapters and which claims remain safe after the audit.

## How this repository demonstrates basic RAG

The basic path is deliberately visible and non-magical:

1. **Ingestion:** extract sections from PDF, DOCX, ODT, Markdown, or text.
2. **Chunking:** split section text at a maximum character size with overlap.
3. **Identity:** attach source filename, title, page when available, and a
   stable chunk ID.
4. **Embedding:** encode each passage with `intfloat/multilingual-e5-large`.
5. **Indexing:** insert normalized 1024-dimensional vectors into FAISS
   `IndexFlatL2`.
6. **Retrieval:** encode the question and request top-K nearest vectors.
7. **Context:** retain candidates in FAISS order for RAG v0, add source headers,
   and enforce a context budget.
8. **Generation:** place the question before the context in a grounded model
   prompt, tokenize the tokenizer's native Qwen chat template, and generate a
   concise answer with the prequantized NF4 model.

Index publication is intentionally separate from indexing: the pipeline first
builds a complete FAISS/mapping generation and only then switches the active
pointer. This lets the book discuss failed updates without implying that the
previous index must be discarded first.

The code exposes `retrieval_results` separately from `search_results`, so the
book can show which passages vector retrieval found and which passages were
ultimately used.

### First educational infographic: basic RAG

The first infographic should communicate these concepts:

- A document is not inserted into an LLM as a whole.
- Extraction and chunking determine what units can be retrieved.
- An embedding maps text to a vector representation.
- FAISS compares vector representations and returns top-K candidates.
- Similarity rank is not a truth or relevance guarantee.
- Source and chunk identity should travel with each retrieved passage.
- Context assembly is a separate stage with explicit limits.
- The generation prompt should distinguish question, evidence, and instruction.
- The output is only as grounded as the retrieved context.

Suggested visual sequence:

```text
Document -> Chunks -> Embeddings -> FAISS -> Top-K -> Context -> LLM
                 \________________ source/chunk metadata ________/
```

Important teaching point from the tests: asking FAISS for more vectors than an
index contains can return `-1`. The application must filter invalid indices
before mapping vectors back to passages.

### Second educational infographic: optional reranking

The second infographic should communicate:

- RAG v0 and RAG v1 share the same ingestion and candidate retrieval.
- RAG v1 does not search a larger corpus.
- A cross-encoder reads the query and one candidate together.
- Cross-encoder scores reorder the already retrieved candidate pool.
- The reranked list changes which evidence is placed first and which evidence
  survives a context budget.
- Reranking cannot recover a passage omitted by the initial vector search.
- Reranking adds compute and model complexity.

Suggested visual sequence:

```text
Question + Corpus
       -> Dense top-K candidate pool
       -> same candidates
          -> RAG v0: vector order
          -> pair each candidate with question
             -> cross-encoder score
             -> RAG v1: score order
       -> context -> LLM
```

The included tests use a deterministic reranker double to demonstrate a
ranking change. The book must label that as a pipeline example, not evidence
that the real MiniLM model improves this corpus.

## Synthetic teaching example

The synthetic corpus has three documents:

- `aster-100.md`: firmware 2.4.0;
- `aster-200.md`: firmware 3.1.0 with similar calibration language;
- `field-calibration.md`: generic calibration instructions.

The example question is:

> Which firmware version does Aster-100 require for calibration?

The intended lesson is broader than the answer:

- `field-calibration.md` is semantically related to the question but does not
  identify the model-specific firmware.
- `aster-200.md` discusses the same concept but states a different model and
  firmware.
- `aster-100.md` contains the specific evidence needed for the answer.
- A retrieval stage can return several plausible passages without proving that
  the first passage is the best evidence.
- A reranker can change ordering when the initial pool already contains the
  correct passage.

The synthetic files explicitly state that they are fictional. The book should
not present their values as real product specifications.

## Safe claims

The book can safely claim that the repository:

- uses 1024-dimensional E5 embeddings, normalized for cosine-equivalent ranking
  in an L2 FAISS index;
- uses deterministic overlapping chunks and structured source metadata;
- filters invalid FAISS IDs;
- supports RAG v0 without reranking and RAG v1 with cross-encoder reranking;
- reorders the same top-K candidate pool in RAG v1;
- passes explicit `(query, passage)` pairs to the cross-encoder;
- preserves the question before retrieved context in the prompt;
- uses tokenizer chat templates when available;
- limits prompt tokens and generated tokens through configuration;
- stores mappings as JSON and validates index compatibility;
- returns no generated answer when retrieval is empty;
- has deterministic tests covering the preceding pipeline contracts.

The book should avoid claiming that:

- the repository implements hybrid retrieval;
- FAISS is a database or a complete metadata store;
- similarity equals factual relevance;
- reranking improves recall;
- the synthetic test demonstrates real-world E5 or MiniLM accuracy;
- an LLM answer is automatically grounded because context was supplied;
- the Flask app is production-ready or secure for public exposure.

## What the basic system cannot solve

### Recall and representation

- Exact identifiers, acronyms, product codes, and rare terms may be better
  served by lexical retrieval.
- One dense vector may not represent multiple facets of a long query.
- Similar wording can dominate over the precise evidence.
- The top-K cutoff can exclude the only answer-bearing passage.

### Chunking and document structure

- Heading heuristics can merge or split sections incorrectly.
- Tables, diagrams, page columns, headers, and scans are not handled reliably.
- Overlap creates duplicated evidence but does not guarantee semantic
  completeness.
- A chunk can lose the surrounding conditions needed to interpret a fact.

### Query understanding

- No query decomposition, rewriting, expansion, clarification, or routing.
- Ambiguous questions may need a profile, domain, source type, or user intent.
- No distinction between exact lookup, broad synthesis, and multi-hop queries.

### Evidence and evaluation

- Source metadata supports traceability but not entailment.
- No claim-level citations, evidence sufficiency score, or contradiction check.
- No labelled retrieval or generation benchmark.
- No monitoring for stale documents or failed retrieval.

### Operations and security

- No public-service authentication, authorization, rate limiting, or TLS.
- No concurrency model for multi-process index updates.
- No production deployment, backup policy, or observability stack.

## How these limitations motivate later chapters

### Hybrid retrieval

The synthetic product/firmware example motivates combining:

- dense embeddings for semantic similarity;
- BM25 for exact identifiers and rare terms;
- SPLADE or other sparse learned representations for lexical-semantic recall;
- metadata filtering for model, version, date, and source type.

### Multi-stage retrieval

The distinction between top-K candidate recall and cross-encoder ordering
motivates:

- larger candidate pools;
- multiple retrieval channels;
- fusion of ranked lists;
- optional reranking;
- token-budget-aware evidence selection.

### Vector infrastructure

FAISS demonstrates vector search, but later chapters can explain Qdrant and
similar systems when persistent filters, payload metadata, updates, and
hosted service semantics are required. This repository should not pretend
FAISS supplies those database features.

### GraphRAG and knowledge graphs

The current chunks contain local prose but no explicit entities or relations.
Questions requiring relationships across passages motivate:

- entity and relation extraction;
- graph expansion;
- multi-hop evidence paths;
- graph-aware ranking and context construction.

### Profiles and adaptive routing

The single fixed pipeline motivates routing by:

- query type;
- source modality;
- latency and cost budget;
- evidence availability;
- required exactness;
- user or organizational profile.

### Evidence-based generation and evaluation

The prompt already distinguishes instruction, question, and source-labelled
context. Later chapters can extend that foundation with:

- evidence units and citations;
- entailment or support checks;
- abstention when evidence is insufficient;
- retrieval recall, precision, and nDCG measurements;
- answer faithfulness and attribution evaluation;
- prompt-injection and document-poisoning tests.

## Recommended chapter sequence

1. Show the synthetic corpus and the question.
2. Follow one document through extraction, chunks, and source metadata.
3. Show the vector index and top-K `retrieval_results`.
4. Build context in RAG v0 order and generate an answer.
5. Repeat with RAG v1 and show a changed ordering when the candidate double
   demonstrates it.
6. State explicitly that only the ordering changed, not the candidate pool.
7. Remove the correct candidate from the pool and show that reranking cannot
   restore it.
8. Use that failure to introduce hybrid recall, multi-stage retrieval,
   GraphRAG, routing, and evidence evaluation.

## Final integration caveat

This repository is a teaching artifact. Its tests validate implementation
contracts and deterministic examples; they are not a retrieval-quality
benchmark. The next architectural chapter should be introduced from the
documented limitations, not by implying that the current pipeline already
contains advanced hybrid RAG behavior.
