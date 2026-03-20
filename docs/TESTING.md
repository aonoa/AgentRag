# Testing Guide

## Phase-by-phase validation

### Phase 0 (scaffold/config/api)

```bash
go test ./...
go build ./...
curl http://localhost:8080/health
```

### Phase 1 (ingestion)

```bash
go test ./internal/ingest -v
curl -X POST http://localhost:8080/v1/ingest/upload -F "file=@RAG开发文档.md"
```

Expected:
- returns `document_id`
- `chunks > 0`
- `summaries = 1`

### Phase 2-4 (ask / retrieval / rerank / grading loop)

```bash
go test ./internal/retrieval ./internal/agent -v
curl -X POST http://localhost:8080/v1/ask -H 'Content-Type: application/json' -d '{"question":"什么是Agentic RAG？","debug":true}'
```

Expected:
- answer is non-empty
- debug includes route and attempt info
