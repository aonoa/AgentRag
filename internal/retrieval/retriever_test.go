package retrieval

import (
	"context"
	"testing"
	"time"

	"agentragplus/internal/config"
	"agentragplus/internal/domain"
	"agentragplus/internal/llm"
	"agentragplus/internal/store"
)

func TestDirectChunkRetrieval(t *testing.T) {
	cfg := config.Config{
		EmbeddingModel:    "mock-embed",
		ChunkCollection:   "chunks",
		SummaryCollection: "summaries",
		TopK:              2,
	}
	ms := store.NewMemoryStore()
	if err := ms.EnsureCollections(context.Background(), cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		t.Fatalf("ensure collections: %v", err)
	}
	if err := ms.UpsertChunks(context.Background(), cfg.ChunkCollection, []domain.Chunk{
		{ChunkID: "c1", DocumentID: "d1", SummaryID: "s1", Source: "doc1", Text: "Go and Eino integration", Embedding: llmVector("Go and Eino integration"), CreatedAt: time.Now()},
		{ChunkID: "c2", DocumentID: "d2", SummaryID: "s2", Source: "doc2", Text: "SQL query optimization", Embedding: llmVector("SQL query optimization"), CreatedAt: time.Now()},
	}); err != nil {
		t.Fatalf("upsert chunks: %v", err)
	}

	r := NewRetriever(cfg, llm.NewMockClient(), ms)
	out, err := r.DirectChunk(context.Background(), "Eino", nil)
	if err != nil {
		t.Fatalf("direct chunk retrieval failed: %v", err)
	}
	if len(out.Candidates) == 0 {
		t.Fatal("expected retrieval candidates")
	}
}

func TestSparseChunkRetrieval(t *testing.T) {
	cfg := config.Config{
		EmbeddingModel:    "mock-embed",
		ChunkCollection:   "chunks",
		SummaryCollection: "summaries",
		TopK:              3,
	}
	ms := store.NewMemoryStore()
	if err := ms.EnsureCollections(context.Background(), cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		t.Fatalf("ensure collections: %v", err)
	}
	if err := ms.UpsertChunks(context.Background(), cfg.ChunkCollection, []domain.Chunk{
		{ChunkID: "c1", DocumentID: "d1", SummaryID: "s1", Source: "doc1", Text: "hybrid retrieval with bm25 sparse and dense vectors", Embedding: llmVector("hybrid retrieval"), CreatedAt: time.Now()},
		{ChunkID: "c2", DocumentID: "d2", SummaryID: "s2", Source: "doc2", Text: "pure vector cosine retrieval", Embedding: llmVector("cosine retrieval"), CreatedAt: time.Now()},
	}); err != nil {
		t.Fatalf("upsert chunks: %v", err)
	}

	r := NewRetriever(cfg, llm.NewMockClient(), ms)
	out, err := r.SparseChunk(context.Background(), "bm25 sparse retrieval", nil)
	if err != nil {
		t.Fatalf("sparse chunk retrieval failed: %v", err)
	}
	if len(out.Candidates) == 0 {
		t.Fatal("expected sparse retrieval candidates")
	}
	if out.Candidates[0].Layer != "sparse_chunk" {
		t.Fatalf("unexpected layer: %s", out.Candidates[0].Layer)
	}
	if out.Candidates[0].ChunkID != "c1" {
		t.Fatalf("expected c1 to rank first for bm25 sparse query, got %s", out.Candidates[0].ChunkID)
	}
}

func llmVector(text string) []float64 {
	vecs, _ := llm.NewMockClient().Embed(context.Background(), "mock", []string{text})
	return vecs[0]
}
