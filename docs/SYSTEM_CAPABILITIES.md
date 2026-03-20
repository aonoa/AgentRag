# AgentRagPlus System Capabilities (Full Summary)

This document is a complete, implementation-aligned summary of the current system: architecture, features, APIs, compatibility, configuration, observability, safety controls, and operations.

---

## 1) System Overview

AgentRagPlus is an **Eino-based Agentic RAG service** built around a workflow-driven orchestration engine.

Primary goals:

- Multi-route intelligent QA (`direct`, `catalog`, `direct_chunk`, `hierarchical`, `hybrid`, `sql`, `web`, `skill`)
- Reliable ingestion pipeline (chunk/summarize/embed/store)
- Retrieval quality via dense+sparse hybrid and RRF fusion
- OpenAI-compatible chat endpoint (including streaming and tool-calling compatibility subset)
- Production-oriented controls (retry, fallback, budget, timeout, observability)

---

## 2) High-Level Architecture

Core modules:

- `cmd/server` — service bootstrap and dependency wiring
- `internal/api` — HTTP API layer (`/health`, ingest, ask, stream, openai-compat)
- `internal/agent` — orchestration, routing, workflow, grading/retry, fallback
- `internal/ingest` — file normalization/chunking/summary/embedding/store write
- `internal/retrieval` — retrieval strategies and summary browsing
- `internal/rerank` — external rerank adapter with graceful fallback
- `internal/llm` — Eino-Ext OpenAI model/embed client + mock client
- `internal/store` — vector store adapter (`memory` + `qdrant` SDK implementation)
- `internal/tools` — SQL/Web external tools
- `internal/skills` — skill registry (`<skill>/SKILL.md`)
- `internal/obs` — tracing/callback/log-correlation helpers

---

## 3) Agent Orchestration Features

### 3.1 Workflow-driven processing

Agent processing is graph/workflow-based (`compose.Workflow`) with node stages:

- `init`
- `rewrite`
- `retrieve`
- `rerank`
- `context`
- `prompt`
- `answer`
- `output`

This replaces manual ad-hoc chaining and provides explicit execution boundaries.

### 3.2 Route types

Supported route enums:

- `direct` — no-retrieval direct answer path
- `catalog` — summary-only browsing path for knowledge overview
- `direct_chunk` — direct dense retrieval
- `hierarchical` — summary->chunk layered retrieval
- `hybrid` — dense + sparse fusion with RRF
- `sql` — SQL tool-based retrieval/lookup
- `web` — external web search
- `skill` — skill-guided answering path

### 3.3 Direct-confidence closed loop (hallucination reduction)

Direct path now includes confidence calibration:

1. Generate direct answer
2. Grade relevance/confidence
3. If below threshold, auto-fallback to lightweight retrieval route (default `catalog`)
4. Re-answer with retrieved context

Configurable via:

- `DIRECT_CONFIDENCE_THRESHOLD`
- `DIRECT_AUTO_FALLBACK`
- `DIRECT_FALLBACK_ROUTE`

### 3.4 Retry and grading loop

- Bounded retry (`MAX_RETRY_LOOPS`)
- Relevance grading per attempt
- Attempt-level debug payload with retrieval/orchestration telemetry

### 3.5 Multi-subquery orchestration

- Query decomposition (`planSubQueries`)
- Per-subquery route execution with local-first and controlled fallback
- Multi-result fusion via RRF (`multi_rrf`)

---

## 4) Retrieval Capabilities

### 4.1 Dense retrieval

- Embedding-based vector search against chunk collection

### 4.2 Sparse retrieval

- BM25 lexical scoring (IDF/avgdl)
- Qdrant sparse-native query when schema supports it
- Compatible fallback for legacy dense-only collections

### 4.3 Hybrid retrieval

- Route `hybrid` executes dense + sparse in parallel semantics
- Fusion algorithm: **RRF** (k=60)

### 4.4 Hierarchical retrieval

- Summary-level recall first
- Expand to chunk-level candidates by summary IDs
- Lightweight rerank-like scoring via vector similarity

### 4.5 Catalog browsing

- `catalog` route performs summary-only retrieval
- Designed for “what content is available?” style queries
- Useful for large, complex knowledge bases before deep retrieval

---

## 5) Ingestion Pipeline

`/v1/ingest/upload` pipeline:

1. File parse/normalize
2. Document ID generation
3. Content class detection (narrative/table)
4. Chunk split (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
5. Summary generation
6. Chunk + summary embedding
7. Upsert into vector store (summary collection + chunk collection)

### Qdrant ID compatibility hardening

- Domain IDs (e.g. `sum_*`, `chk_*`) are preserved in payload
- Qdrant point IDs are deterministic UUID-compatible values
- Retrieval converts back to payload business IDs for downstream consistency

---

## 6) Skill Capability

Skill mode is supported via `SKILLS_DIR`:

- Directory convention: `<SKILLS_DIR>/<skillName>/SKILL.md`
- Optional YAML frontmatter parsed (`name`, `description`)
- Request can explicitly set `skill` + `skill_args`
- If explicit skill is provided, behavior is deterministic (resolve or error)
- Optional auto-selection by question/name match

Security and consistency hardening:

- Skill name validation (`^[a-z0-9_-]+$`)
- Directory traversal/symlink boundary checks
- Ask/AskStream behavior parity for missing/invalid skill

---

## 7) OpenAI Compatibility Layer

Endpoint: `POST /v1/chat/completions`

### 7.1 Supported compatibility features

- Non-stream and stream response formats
- OpenAI-like error envelope
- Message content extraction from string and common array/object text forms
- Approximate `usage` fields

### 7.2 Tool-calling compatibility (implemented subset)

- Accepts `tools` and `tool_choice` (`auto/none/required` + function target)
- Returns `finish_reason=tool_calls` with `message.tool_calls` (non-stream)
- Emits `delta.tool_calls` and final tool_calls finish chunk (stream)
- Supports follow-up `role=tool` messages feeding final answer generation
- Enforces stricter `tool_choice=required` semantics (no silent downgrade)
- Validates missing tool outputs for prior assistant tool-calls

---

## 8) Configuration System

### 8.1 Load precedence

1. Built-in defaults (code)
2. Base config file (`config/defaults.json` or `CONFIG_FILE`)
3. Environment overlay (`config/environments/<APP_ENV>.json`)
4. Environment variables (highest priority)

### 8.2 Structured config groups

- `app`
- `llm`
- `embedding`
- `timeouts`
- `vector`
- `retrieval`
- `rerank`
- `routing`
- `sql`
- `web`
- `skills`
- `observability`
- `orchestration`

### 8.3 Split model/embed endpoints and keys

URLs:

- `OPENAI_LLM_BASE_URL`
- `OPENAI_EMBEDDING_BASE_URL`
- `OPENAI_BASE_URL` (fallback)

Keys:

- `OPENAI_LLM_API_KEY`
- `OPENAI_EMBEDDING_API_KEY`
- `OPENAI_API_KEY` (fallback)

---

## 9) Store Adapter and Backend Strategy

### 9.1 Adapter interface

Business logic depends on `VectorStore` interface, not concrete backend.

Current backends:

- `memory` (test/dev)
- `qdrant` (`github.com/qdrant/go-client`)

### 9.2 SQL backend flexibility

SQL tool supports configurable drivers:

- `sqlite`
- `pgx`
- `mysql`
- `sqlserver`

---

## 10) Observability (Chapter 06 aligned)

### 10.1 Trace + callback foundation

- Global callback handler registration (guarded register-once)
- Request and workflow correlation IDs in context
- Span boundaries at API/agent/workflow/LLM/retrieval/ingest/tools/rerank

### 10.2 Correlation contract

- `request_id`
- `workflow_run_id`
- `trace_id`
- `span_id`

### 10.3 API operational visibility

- `X-Request-Id` response header
- Key stream write errors include full trace correlation fields

### 10.4 Observability toggles

- `OBS_ENABLE_TRACING`
- `OBS_ENABLE_CALLBACKS`

---

## 11) Safety and Guardrails

- SQL tool only allows single `SELECT` and blocks multi-statement execution
- Bounded retries and orchestrator timeout
- External call cap for SQL/Web fallback
- Early-stop thresholds to reduce unnecessary cost/latency
- Skill/path validation and deterministic explicit-skill behavior

---

## 12) Primary HTTP Endpoints

- `GET /health`
- `POST /v1/ingest/upload`
- `POST /v1/ask`
- `POST /v1/ask/stream`
- `POST /v1/chat/completions` (OpenAI-compatible layer)

---

## 13) Runtime and Delivery

- Dockerfile + docker-compose ready
- Makefile profiles for `default/local/dev/staging/prod`
- Go test/build pipeline verified

---

## 14) Current Compatibility Scope (Practical)

The system is production-oriented and feature-rich, but OpenAI compatibility is still a **well-defined subset** (especially around advanced multi-round tool orchestration semantics). Core SDK interoperability for common chat/stream/tool-call flows is supported.

---

## 15) Suggested Next Enhancements

- Strict multi-round tool-call state machine persistence
- Trace sampling and exporter strategy tuning
- Optional sensitive-field redaction policy for observability events
- Broader OpenAI response-format/tool schema parity tests
