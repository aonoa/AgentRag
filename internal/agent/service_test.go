package agent

import (
	"context"
	"os"
	"path/filepath"
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
	if _, ok := directMap["grade"]; !ok {
		t.Fatal("expected grade in direct debug map")
	}
}

func TestAskCatalogRouteReturnsSummaryContext(t *testing.T) {
	cfg := config.Config{
		LLMModel:          "mock-llm",
		EmbeddingModel:    "mock-embed",
		RouterModel:       "mock-llm",
		GradeModel:        "mock-llm",
		TopK:              4,
		RerankTopM:        4,
		MaxRetries:        1,
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
	if err := ms.UpsertSummaries(context.Background(), cfg.SummaryCollection, []domain.Summary{
		{SummaryID: "sum_a", DocumentID: "doc_a", Source: "kb_a.md", Text: "A 文档讲了 Agentic RAG 架构概览", Embedding: embedVec(t, llmClient, "Agentic RAG 架构概览")},
		{SummaryID: "sum_b", DocumentID: "doc_b", Source: "kb_b.md", Text: "B 文档讲了向量检索和混合检索", Embedding: embedVec(t, llmClient, "向量检索和混合检索")},
	}); err != nil {
		t.Fatalf("upsert summaries: %v", err)
	}

	retr := retrieval.NewRetriever(cfg, llmClient, ms)
	rrk := rerank.NewHTTPClient(cfg)
	sqlTool := tools.NewSQLTool(nil, llmClient, cfg)
	webTool := tools.NewWebTool(cfg)
	svc, err := NewService(cfg, llmClient, retr, rrk, sqlTool, webTool)
	if err != nil {
		t.Fatalf("new service: %v", err)
	}

	out, err := svc.Ask(context.Background(), domain.AskRequest{Question: "先看看知识库目录有哪些内容", ForceRoute: domain.RouteCatalog, Debug: true})
	if err != nil {
		t.Fatalf("ask catalog failed: %v", err)
	}
	if out.Route != domain.RouteCatalog {
		t.Fatalf("expected catalog route, got %s", out.Route)
	}
	if out.Attempts != 1 {
		t.Fatalf("expected one attempt, got %d", out.Attempts)
	}
	if len(out.References) == 0 {
		t.Fatal("expected catalog route to return summary references")
	}
	if out.Debug == nil {
		t.Fatal("expected debug payload")
	}
}

func TestAskExplicitSkill(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "eino-guide")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatalf("mkdir skill dir: %v", err)
	}
	skillDoc := "---\nname: eino-guide\ndescription: Eino 入门指南\n---\n这是一个 Eino 入门技能文档。"
	if err := os.WriteFile(filepath.Join(skillDir, "SKILL.md"), []byte(skillDoc), 0o644); err != nil {
		t.Fatalf("write skill doc: %v", err)
	}

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
		SkillsDir:         dir,
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

	out, err := svc.Ask(context.Background(), domain.AskRequest{Question: "请用eino-guide回答", Skill: "eino-guide", Debug: true})
	if err != nil {
		t.Fatalf("ask skill failed: %v", err)
	}
	if out.Route != domain.RouteSkill {
		t.Fatalf("expected skill route, got %s", out.Route)
	}
	if len(out.References) == 0 {
		t.Fatal("expected skill references")
	}
	if out.Debug == nil {
		t.Fatal("expected skill debug payload")
	}
}

func TestAskExplicitSkillNotFound(t *testing.T) {
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

	if _, err := svc.Ask(context.Background(), domain.AskRequest{Question: "test", Skill: "eino-guide"}); err == nil {
		t.Fatal("expected explicit missing skill error")
	}
}

func TestAskStreamExplicitSkillNotFound(t *testing.T) {
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

	if _, err := svc.AskStream(context.Background(), domain.AskRequest{Question: "test", Skill: "eino-guide"}); err == nil {
		t.Fatal("expected explicit missing skill error")
	}
}

func embedVec(t *testing.T, c llm.Client, text string) []float64 {
	t.Helper()
	vecs, err := c.Embed(context.Background(), "mock-embed", []string{text})
	if err != nil {
		t.Fatalf("embed failed: %v", err)
	}
	if len(vecs) == 0 {
		t.Fatal("empty embedding output")
	}
	return vecs[0]
}
