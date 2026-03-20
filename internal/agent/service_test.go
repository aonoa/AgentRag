package agent

import (
	"context"
	"testing"

	"agentragplus/internal/config"
	"agentragplus/internal/domain"
	"agentragplus/internal/ingest"
	"agentragplus/internal/llm"
	"agentragplus/internal/rerank"
	"agentragplus/internal/retrieval"
	"agentragplus/internal/store"
	"agentragplus/internal/tools"
)

func TestMergeCandidatesRRF(t *testing.T) {
	dense := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{
		{ChunkID: "A", Score: 0.9},
		{ChunkID: "B", Score: 0.8},
		{ChunkID: "C", Score: 0.7},
	}}
	sparse := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{
		{ChunkID: "B", Score: 2.0},
		{ChunkID: "A", Score: 1.8},
		{ChunkID: "D", Score: 1.5},
	}}

	merged := mergeCandidatesRRF(dense, sparse, 10)
	if len(merged.Candidates) == 0 {
		t.Fatal("expected merged candidates")
	}
	if merged.Debug["mode"] != "hybrid_rrf" {
		t.Fatalf("expected hybrid_rrf mode, got %v", merged.Debug["mode"])
	}
	if merged.Candidates[0].ChunkID != "A" && merged.Candidates[0].ChunkID != "B" {
		t.Fatalf("unexpected top candidate: %s", merged.Candidates[0].ChunkID)
	}
}

func TestAskFlow(t *testing.T) {
	cfg := config.Config{
		LLMModel:          "mock-llm",
		EmbeddingModel:    "mock-embed",
		RouterModel:       "mock-llm",
		GradeModel:        "mock-llm",
		TopK:              4,
		RerankTopM:        2,
		MaxRetries:        2,
		ChunkSize:         40,
		ChunkOverlap:      10,
		ChunkCollection:   "chunks",
		SummaryCollection: "summaries",
	}
	llmClient := llm.NewMockClient()
	ms := store.NewMemoryStore()
	if err := ms.EnsureCollections(context.Background(), cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		t.Fatalf("ensure collections: %v", err)
	}
	_ingest := ingest.NewService(cfg, llmClient, ms)
	if _, err := _ingest.IngestFile(context.Background(), "kb.md", []byte("Eino 是一个用于构建智能体与工作流编排的框架。")); err != nil {
		t.Fatalf("ingest: %v", err)
	}

	retr := retrieval.NewRetriever(cfg, llmClient, ms)
	rrk := rerank.NewHTTPClient(cfg)
	sqlTool := tools.NewSQLTool(nil, llmClient, cfg)
	webTool := tools.NewWebTool(cfg)
	svc, err := NewService(cfg, llmClient, retr, rrk, sqlTool, webTool)
	if err != nil {
		t.Fatalf("new service: %v", err)
	}

	out, err := svc.Ask(context.Background(), domain.AskRequest{Question: "什么是Eino？", Debug: true})
	if err != nil {
		t.Fatalf("ask failed: %v", err)
	}
	if out.Answer == "" {
		t.Fatal("empty answer")
	}
	if out.Attempts < 1 {
		t.Fatalf("unexpected attempts: %d", out.Attempts)
	}
}

func TestSplitByPunct(t *testing.T) {
	parts := splitByPunct("问题A，问题B；问题C？")
	if len(parts) < 3 {
		t.Fatalf("expected >= 3 parts, got %d", len(parts))
	}
}

func TestFuseMultipleResultsRRF(t *testing.T) {
	r1 := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "a", Score: 1}, {ChunkID: "b", Score: 0.9}}}
	r2 := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "b", Score: 1}, {ChunkID: "c", Score: 0.8}}}
	r3 := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "a", Score: 1}, {ChunkID: "c", Score: 0.9}}}

	out := fuseMultipleResultsRRF([]domain.RetrievalResult{r1, r2, r3}, 5)
	if len(out.Candidates) == 0 {
		t.Fatal("expected fused candidates")
	}
	if out.Debug["mode"] != "multi_rrf" {
		t.Fatalf("unexpected mode: %v", out.Debug["mode"])
	}
}

func TestShouldEarlyStop(t *testing.T) {
	res := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "a", Score: 0.1}, {ChunkID: "b", Score: 0.05}, {ChunkID: "c", Score: 0.01}}}
	if !shouldEarlyStop(res, 3, 0) {
		t.Fatal("expected early stop by min candidates")
	}
	if !shouldEarlyStop(res, 10, 0.08) {
		t.Fatal("expected early stop by top score")
	}
	if shouldEarlyStop(domain.RetrievalResult{}, 1, 0.1) {
		t.Fatal("did not expect early stop for empty results")
	}
}

func TestAskDirectRouteSkipsRetrievalPipeline(t *testing.T) {
	cfg := config.Config{
		LLMModel:          "mock-llm",
		EmbeddingModel:    "mock-embed",
		RouterModel:       "mock-llm",
		GradeModel:        "mock-llm",
		TopK:              4,
		RerankTopM:        2,
		MaxRetries:        2,
		ChunkSize:         40,
		ChunkOverlap:      10,
		ChunkCollection:   "chunks",
		SummaryCollection: "summaries",
	}
	llmClient := llm.NewMockClient()
	ms := store.NewMemoryStore()
	if err := ms.EnsureCollections(context.Background(), cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		t.Fatalf("ensure collections: %v", err)
	}
	retr := retrieval.NewRetriever(cfg, llmClient, ms)
	rrk := rerank.NewHTTPClient(cfg)
	sqlTool := tools.NewSQLTool(nil, llmClient, cfg)
	webTool := tools.NewWebTool(cfg)
	svc, err := NewService(cfg, llmClient, retr, rrk, sqlTool, webTool)
	if err != nil {
		t.Fatalf("new service: %v", err)
	}

	out, err := svc.Ask(context.Background(), domain.AskRequest{Question: "你好", ForceRoute: domain.RouteDirect, Debug: true})
	if err != nil {
		t.Fatalf("ask direct failed: %v", err)
	}
	if out.Route != domain.RouteDirect {
		t.Fatalf("expected direct route, got %s", out.Route)
	}
	if out.Attempts != 1 {
		t.Fatalf("expected one attempt for direct no-retrieval, got %d", out.Attempts)
	}
	if len(out.References) != 0 {
		t.Fatalf("expected no references for direct no-retrieval, got %d", len(out.References))
	}
	directRaw, ok := out.Debug["direct"]
	if !ok {
		t.Fatal("expected direct debug block")
	}
	directMap, ok := directRaw.(map[string]any)
	if !ok {
		t.Fatal("expected direct debug map")
	}
	if skipped, ok := directMap["retrieval_skipped"].(bool); !ok || !skipped {
		t.Fatal("expected retrieval_skipped=true")
	}
}
