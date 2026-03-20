package rerank

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"agentragplus/internal/config"
	"agentragplus/internal/domain"
)

type Client interface {
	Rerank(ctx context.Context, query string, candidates []domain.RetrievalCandidate, topM int) ([]domain.RetrievalCandidate, error)
}

type HTTPClient struct {
	cfg   config.Config
	httpc *http.Client
}

func NewHTTPClient(cfg config.Config) *HTTPClient {
	return &HTTPClient{
		cfg:   cfg,
		httpc: &http.Client{Timeout: cfg.HTTPTimeout},
	}
}

func (c *HTTPClient) Rerank(ctx context.Context, query string, candidates []domain.RetrievalCandidate, topM int) ([]domain.RetrievalCandidate, error) {
	if len(candidates) == 0 {
		return candidates, nil
	}
	if topM <= 0 {
		topM = len(candidates)
	}
	if c.cfg.RerankURL == "" || c.cfg.RerankKey == "" {
		return topByScore(candidates, topM), nil
	}

	documents := make([]string, 0, len(candidates))
	for _, item := range candidates {
		documents = append(documents, item.Text)
	}
	reqBody := map[string]any{
		"model":     c.cfg.RerankModel,
		"query":     query,
		"documents": documents,
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal rerank request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.RerankURL, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("create rerank request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.cfg.RerankKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send rerank request: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("rerank request failed status=%d body=%s", resp.StatusCode, string(body))
	}

	var out struct {
		Results []struct {
			Index int     `json:"index"`
			Score float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("unmarshal rerank response: %w", err)
	}
	rescored := make([]domain.RetrievalCandidate, 0, len(out.Results))
	for _, row := range out.Results {
		if row.Index < 0 || row.Index >= len(candidates) {
			continue
		}
		item := candidates[row.Index]
		item.Score = row.Score
		rescored = append(rescored, item)
	}
	if len(rescored) == 0 {
		return topByScore(candidates, topM), nil
	}
	if len(rescored) > topM {
		rescored = rescored[:topM]
	}
	return rescored, nil
}

func topByScore(cands []domain.RetrievalCandidate, topM int) []domain.RetrievalCandidate {
	dup := make([]domain.RetrievalCandidate, len(cands))
	copy(dup, cands)
	sort.Slice(dup, func(i, j int) bool { return dup[i].Score > dup[j].Score })
	if len(dup) > topM {
		dup = dup[:topM]
	}
	for i := range dup {
		dup[i].Layer = strings.TrimSpace(dup[i].Layer)
	}
	return dup
}
