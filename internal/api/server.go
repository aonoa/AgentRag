package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"agentragplus/internal/agent"
	"agentragplus/internal/config"
	"agentragplus/internal/domain"
	"agentragplus/internal/ingest"
)

type Server struct {
	cfg      config.Config
	logger   *slog.Logger
	ingest   *ingest.Service
	agentSvc *agent.Service
	mux      *http.ServeMux
}

func NewServer(cfg config.Config, logger *slog.Logger, ingestSvc *ingest.Service, agentSvc *agent.Service) *Server {
	s := &Server{
		cfg:      cfg,
		logger:   logger,
		ingest:   ingestSvc,
		agentSvc: agentSvc,
		mux:      http.NewServeMux(),
	}
	s.routes()
	return s
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) routes() {
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/v1/ingest/upload", s.handleUpload)
	s.mux.HandleFunc("/v1/ask", s.handleAsk)
	s.mux.HandleFunc("/v1/ask/stream", s.handleAskStream)
	s.mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"service": "agentragplus",
	})
}

func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	if err := r.ParseMultipartForm(20 << 20); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid multipart form"})
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing file field 'file'"})
		return
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "read file failed"})
		return
	}

	out, err := s.ingest.IngestFile(r.Context(), header.Filename, content)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleAsk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req domain.AskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json body"})
		return
	}
	req.Question = strings.TrimSpace(req.Question)
	if req.Question == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "question is required"})
		return
	}

	out, err := s.agentSvc.Ask(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleAskStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	var req domain.AskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json body"})
		return
	}
	req.Question = strings.TrimSpace(req.Question)
	if req.Question == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "question is required"})
		return
	}

	out, err := s.agentSvc.AskStream(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	if err := streamAskResponse(w, out); err != nil {
		s.logger.Error("ask stream write failed", "error", err.Error())
	}
}

type openAIChatRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Temperature *float64 `json:"temperature,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
	User        string   `json:"user,omitempty"`
}

type openAIChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type openAIErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeOpenAIError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}

	var req openAIChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid json body")
		return
	}
	if len(req.Messages) == 0 {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "messages is required")
		return
	}
	question := extractUserMessage(req.Messages)
	if question == "" {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "at least one non-empty user message is required")
		return
	}

	askReq := domain.AskRequest{
		Question: question,
		Debug:    false,
	}
	model := chooseModel(req.Model, s.cfg.LLMModel)

	if req.Stream {
		streamResult, err := s.agentSvc.AskStream(r.Context(), askReq)
		if err != nil {
			writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		if err := streamOpenAIChatCompletions(w, model, streamResult); err != nil {
			s.logger.Error("openai stream write failed", "error", err.Error())
		}
		return
	}

	out, err := s.agentSvc.Ask(r.Context(), askReq)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	resp := openAIChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
	}
	resp.Choices = []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}{
		{
			Index: 0,
			Message: struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			}{
				Role:    "assistant",
				Content: out.Answer,
			},
			FinishReason: "stop",
		},
	}

	promptTokens := estimateTokens(question)
	completionTokens := estimateTokens(out.Answer)
	resp.Usage.PromptTokens = promptTokens
	resp.Usage.CompletionTokens = completionTokens
	resp.Usage.TotalTokens = promptTokens + completionTokens

	writeJSON(w, http.StatusOK, resp)
}

func extractUserMessage(messages []struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if strings.EqualFold(strings.TrimSpace(messages[i].Role), "user") {
			text := strings.TrimSpace(messages[i].Content)
			if text != "" {
				return text
			}
		}
	}
	return ""
}

func chooseModel(requestModel string, fallback string) string {
	m := strings.TrimSpace(requestModel)
	if m != "" {
		return m
	}
	return fallback
}

func estimateTokens(text string) int {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return 0
	}
	runes := []rune(trimmed)
	tokens := len(runes) / 4
	if len(runes)%4 != 0 {
		tokens++
	}
	if tokens < 1 {
		return 1
	}
	return tokens
}

func writeOpenAIError(w http.ResponseWriter, status int, typ string, message string) {
	errResp := openAIErrorResponse{}
	errResp.Error.Type = typ
	errResp.Error.Message = message
	errResp.Error.Code = ""
	writeJSON(w, status, errResp)
}

func streamAskResponse(w http.ResponseWriter, out agent.AskStreamResult) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("streaming not supported by response writer")
	}
	defer out.Stream.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	for {
		token, err := out.Stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		payload := map[string]any{"event": "delta", "token": token}
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		if _, err := io.WriteString(w, "data: "+string(data)+"\n\n"); err != nil {
			return err
		}
		flusher.Flush()
	}

	finalPayload := map[string]any{
		"event": "done",
		"route": out.Route,
		"meta": map[string]any{
			"attempts":           out.Attempts,
			"rewritten_question": out.RewrittenQuestion,
		},
	}
	finalData, err := json.Marshal(finalPayload)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: "+string(finalData)+"\n\n"); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func streamOpenAIChatCompletions(w http.ResponseWriter, model string, streamResult agent.AskStreamResult) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return fmt.Errorf("streaming not supported by response writer")
	}
	defer streamResult.Stream.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	streamID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	firstChunk := map[string]any{
		"id":      streamID,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{
			{"index": 0, "delta": map[string]any{"role": "assistant"}, "finish_reason": nil},
		},
	}
	if err := writeSSEJSON(w, firstChunk); err != nil {
		return err
	}
	flusher.Flush()

	for {
		token, err := streamResult.Stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		chunk := map[string]any{
			"id":      streamID,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{"content": token}, "finish_reason": nil},
			},
		}
		if err := writeSSEJSON(w, chunk); err != nil {
			return err
		}
		flusher.Flush()
	}

	lastChunk := map[string]any{
		"id":      streamID,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{
			{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"},
		},
	}
	if err := writeSSEJSON(w, lastChunk); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func writeSSEJSON(w http.ResponseWriter, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = io.WriteString(w, "data: "+string(data)+"\n\n")
	return err
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
