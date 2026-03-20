package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"agentragplus/internal/config"
	"agentragplus/internal/obs"
)

type WebTool struct {
	cfg   config.Config
	httpc *http.Client
}

func NewWebTool(cfg config.Config) *WebTool {
	return &WebTool{cfg: cfg, httpc: &http.Client{Timeout: cfg.HTTPTimeout}}
}

func (t *WebTool) Search(ctx context.Context, query string) (string, error) {
	ctx, span := obs.StartSpan(ctx, "tool.web.search")
	defer obs.EndSpan(span, nil)
	obs.EmitEvent(ctx, "tool.web.search.start")
	if t.cfg.SerperAPIKey == "" {
		return "", fmt.Errorf("web search not configured")
	}
	reqBody := map[string]any{"q": query, "num": 5}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal serper request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://google.serper.dev/search", bytes.NewReader(data))
	if err != nil {
		return "", fmt.Errorf("create serper request: %w", err)
	}
	req.Header.Set("X-API-KEY", t.cfg.SerperAPIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := t.httpc.Do(req)
	if err != nil {
		return "", fmt.Errorf("send serper request: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("serper request failed status=%d body=%s", resp.StatusCode, string(body))
	}

	var out struct {
		Organic []struct {
			Title   string `json:"title"`
			Link    string `json:"link"`
			Snippet string `json:"snippet"`
		} `json:"organic"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return "", fmt.Errorf("unmarshal serper response: %w", err)
	}

	parts := make([]string, 0, len(out.Organic))
	for i, row := range out.Organic {
		parts = append(parts, fmt.Sprintf("[%d] %s\n%s\n%s", i+1, strings.TrimSpace(row.Title), strings.TrimSpace(row.Snippet), strings.TrimSpace(row.Link)))
	}
	if len(parts) == 0 {
		obs.EmitEvent(ctx, "tool.web.search.done")
		return "(no web results)", nil
	}
	obs.EmitEvent(ctx, "tool.web.search.done")
	return strings.Join(parts, "\n\n"), nil
}
