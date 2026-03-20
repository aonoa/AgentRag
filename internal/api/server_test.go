package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"agentragplus/internal/agent"
	"agentragplus/internal/config"
	"agentragplus/internal/ingest"
	"agentragplus/internal/llm"
	"agentragplus/internal/rerank"
	"agentragplus/internal/retrieval"
	"agentragplus/internal/store"
	"agentragplus/internal/tools"
)

func buildTestServer(t *testing.T) *Server {
	t.Helper()
	cfg := config.Config{
		ServerAddr:        ":8081",
		LogLevel:          "debug",
		LLMModel:          "mock-llm",
		EmbeddingModel:    "mock-embed",
		RouterModel:       "mock-llm",
		GradeModel:        "mock-llm",
		TopK:              4,
		RerankTopM:        2,
		MaxRetries:        2,
		ChunkSize:         200,
		ChunkOverlap:      40,
		ChunkCollection:   "chunks",
		SummaryCollection: "summaries",
		UseOpenAI:         false,
		HTTPTimeout:       30,
	}
	logger := slog.New(slog.NewTextHandler(bytes.NewBuffer(nil), nil))
	llmClient := llm.NewMockClient()
	ms := store.NewMemoryStore()
	if err := ms.EnsureCollections(context.Background(), cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		t.Fatalf("ensure collections: %v", err)
	}
	ing := ingest.NewService(cfg, llmClient, ms)
	retr := retrieval.NewRetriever(cfg, llmClient, ms)
	rrk := rerank.NewHTTPClient(cfg)
	sqlTool := tools.NewSQLTool(nil, llmClient, cfg)
	webTool := tools.NewWebTool(cfg)
	agSvc, err := agent.NewService(cfg, llmClient, retr, rrk, sqlTool, webTool)
	if err != nil {
		t.Fatalf("new agent service: %v", err)
	}
	return NewServer(cfg, logger, ing, agSvc)
}

func TestChatCompletionsRequiresModel(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "model is required") {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}

func TestChatCompletionsSupportsArrayContent(t *testing.T) {
	srv := buildTestServer(t)
	payload := map[string]any{
		"model": "mock-llm",
		"messages": []map[string]any{
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "text", "text": "请解释一下 Eino"},
				},
			},
		},
	}
	b, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "chat.completion") {
		t.Fatalf("unexpected response: %s", rec.Body.String())
	}
}

func TestChatCompletionsStreamDone(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	resp := rec.Body.String()
	if !strings.Contains(resp, "chat.completion.chunk") {
		t.Fatalf("expected chunk payloads, got: %s", resp)
	}
	if !strings.Contains(resp, "data: [DONE]") {
		t.Fatalf("expected [DONE], got: %s", resp)
	}
}

func TestChatCompletionsToolCallNonStream(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","tool_choice":"required","tools":[{"type":"function","function":{"name":"search_web"}}],"messages":[{"role":"user","content":"请用search_web查一下 Eino"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	resp := rec.Body.String()
	if !strings.Contains(resp, "\"finish_reason\":\"tool_calls\"") {
		t.Fatalf("expected finish_reason=tool_calls, got: %s", resp)
	}
	if !strings.Contains(resp, "\"tool_calls\"") {
		t.Fatalf("expected tool_calls payload, got: %s", resp)
	}
}

func TestChatCompletionsToolCallStream(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","stream":true,"tool_choice":"required","tools":[{"type":"function","function":{"name":"search_web"}}],"messages":[{"role":"user","content":"请用search_web查一下 Eino"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	resp := rec.Body.String()
	if !strings.Contains(resp, "\"finish_reason\":\"tool_calls\"") {
		t.Fatalf("expected tool_calls finish_reason in stream, got: %s", resp)
	}
	if !strings.Contains(resp, "data: [DONE]") {
		t.Fatalf("expected [DONE], got: %s", resp)
	}
}

func TestChatCompletionsToolResultFollowUp(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","messages":[{"role":"user","content":"请查Eino"},{"role":"assistant","content":""},{"role":"tool","tool_call_id":"call_1","content":"Eino 是一个 Go AI 应用开发框架"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "chat.completion") {
		t.Fatalf("unexpected response: %s", rec.Body.String())
	}
}

func TestChatCompletionsToolChoiceRequiredNoToolSelected(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","tool_choice":"required","tools":[],"messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "tool_choice requires") {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}

func TestChatCompletionsToolCallRequiredReturnsMultipleCalls(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","tool_choice":"required","tools":[{"type":"function","function":{"name":"search_web"}},{"type":"function","function":{"name":"query_sql"}}],"messages":[{"role":"user","content":"请同时用search_web和query_sql查一下"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if strings.Count(rec.Body.String(), "\"type\":\"function\"") < 2 {
		t.Fatalf("expected at least two function tool calls, got: %s", rec.Body.String())
	}
}

func TestChatCompletionsMissingToolOutputsReturnsError(t *testing.T) {
	srv := buildTestServer(t)
	body := `{"model":"mock-llm","messages":[{"role":"user","content":"请查Eino"},{"role":"assistant","content":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_web","arguments":"{\"query\":\"Eino\"}"}}]}}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "missing tool outputs") {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}
