package ingest

import (
	"context"
	"testing"

	"agentragplus/internal/config"
	"agentragplus/internal/llm"
	"agentragplus/internal/store"
)

func testConfig() config.Config {
	return config.Config{
		ChunkSize:         40,
		ChunkOverlap:      10,
		EmbeddingModel:    "mock-embed",
		LLMModel:          "mock-llm",
		ChunkCollection:   "chunks",
		SummaryCollection: "summaries",
	}
}

func TestIngestFile(t *testing.T) {
	cfg := testConfig()
	ms := store.NewMemoryStore()
	if err := ms.EnsureCollections(context.Background(), cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		t.Fatalf("ensure collections: %v", err)
	}
	svc := NewService(cfg, llm.NewMockClient(), ms)

	out, err := svc.IngestFile(context.Background(), "sample.md", []byte("这是一个测试文档。\n它描述了RAG系统。\n用于检验切块与入库流程。"))
	if err != nil {
		t.Fatalf("ingest failed: %v", err)
	}
	if out.DocumentID == "" {
		t.Fatal("expected non-empty document id")
	}
	if out.Chunks == 0 {
		t.Fatal("expected chunks > 0")
	}
	if out.Summaries != 1 {
		t.Fatalf("expected one summary, got %d", out.Summaries)
	}
}

func TestSplitByRunes(t *testing.T) {
	parts := splitByRunes("abcdefghijk", 4, 1)
	if len(parts) < 3 {
		t.Fatalf("expected at least 3 parts, got %d", len(parts))
	}
	if parts[0] != "abcd" {
		t.Fatalf("unexpected first chunk: %q", parts[0])
	}
}
