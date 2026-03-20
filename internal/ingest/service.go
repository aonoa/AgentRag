package ingest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"agentragplus/internal/config"
	"agentragplus/internal/domain"
	"agentragplus/internal/llm"
	"agentragplus/internal/store"
)

type Service struct {
	cfg   config.Config
	llm   llm.Client
	store store.VectorStore
}

func NewService(cfg config.Config, llmClient llm.Client, vectorStore store.VectorStore) *Service {
	return &Service{cfg: cfg, llm: llmClient, store: vectorStore}
}

func (s *Service) IngestFile(ctx context.Context, filename string, raw []byte) (domain.UploadResponse, error) {
	if strings.TrimSpace(filename) == "" {
		return domain.UploadResponse{}, errors.New("filename is required")
	}
	if len(raw) == 0 {
		return domain.UploadResponse{}, errors.New("empty file")
	}

	text, err := normalizeText(filename, raw)
	if err != nil {
		return domain.UploadResponse{}, err
	}
	documentID := hashID(filename + "\n" + text)
	contentClass := detectContentClass(text)

	chunksText := splitByRunes(text, s.cfg.ChunkSize, s.cfg.ChunkOverlap)
	if len(chunksText) == 0 {
		return domain.UploadResponse{}, errors.New("no chunks produced")
	}
	summaryText, err := s.buildSummary(ctx, text, contentClass)
	if err != nil {
		return domain.UploadResponse{}, err
	}

	chunkEmbeddings, err := s.llm.Embed(ctx, s.cfg.EmbeddingModel, chunksText)
	if err != nil {
		return domain.UploadResponse{}, fmt.Errorf("embed chunks: %w", err)
	}
	summaryEmbeddings, err := s.llm.Embed(ctx, s.cfg.EmbeddingModel, []string{summaryText})
	if err != nil {
		return domain.UploadResponse{}, fmt.Errorf("embed summary: %w", err)
	}

	now := time.Now().UTC()
	summaryID := "sum_" + documentID
	summaries := []domain.Summary{{
		SummaryID:    summaryID,
		DocumentID:   documentID,
		Source:       filename,
		Text:         summaryText,
		Embedding:    summaryEmbeddings[0],
		CreatedAt:    now,
		ContentClass: contentClass,
	}}

	chunks := make([]domain.Chunk, 0, len(chunksText))
	for i, ch := range chunksText {
		chunks = append(chunks, domain.Chunk{
			ChunkID:      fmt.Sprintf("chk_%s_%03d", documentID, i),
			DocumentID:   documentID,
			SummaryID:    summaryID,
			Source:       filename,
			Text:         ch,
			Embedding:    chunkEmbeddings[i],
			CreatedAt:    now,
			ChunkIndex:   i,
			ChunkCount:   len(chunksText),
			ContentClass: contentClass,
		})
	}

	if err := s.store.UpsertSummaries(ctx, s.cfg.SummaryCollection, summaries); err != nil {
		return domain.UploadResponse{}, fmt.Errorf("store summaries: %w", err)
	}
	if err := s.store.UpsertChunks(ctx, s.cfg.ChunkCollection, chunks); err != nil {
		return domain.UploadResponse{}, fmt.Errorf("store chunks: %w", err)
	}

	return domain.UploadResponse{
		DocumentID: documentID,
		Filename:   filename,
		Chunks:     len(chunks),
		Summaries:  len(summaries),
	}, nil
}

func (s *Service) buildSummary(ctx context.Context, text string, contentClass string) (string, error) {
	if contentClass == "table" {
		return limitRunes(text, 1200), nil
	}
	prompt := "请将以下文档总结成 5 句以内的知识摘要，保留实体、数字、结论。"
	summary, err := s.llm.Chat(ctx, s.cfg.LLMModel, "INGEST_SUMMARY", prompt+"\n\n"+limitRunes(text, 6000))
	if err != nil {
		return "", fmt.Errorf("llm summarize: %w", err)
	}
	return strings.TrimSpace(summary), nil
}

func normalizeText(filename string, raw []byte) (string, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	if !utf8.Valid(raw) {
		return "", fmt.Errorf("unsupported non-utf8 file: %s", ext)
	}
	text := strings.TrimSpace(string(raw))
	if text == "" {
		return "", errors.New("file has no readable text")
	}
	return text, nil
}

func detectContentClass(text string) string {
	lines := strings.Split(text, "\n")
	if len(lines) < 2 {
		return "narrative"
	}
	tabularHits := 0
	for _, line := range lines {
		trim := strings.TrimSpace(line)
		if trim == "" {
			continue
		}
		if strings.Count(trim, "|") >= 2 || strings.Count(trim, ",") >= 2 || strings.Count(trim, "\t") >= 2 {
			tabularHits++
		}
	}
	threshold := 2
	if len(lines)/3 > threshold {
		threshold = len(lines) / 3
	}
	if tabularHits >= threshold {
		return "table"
	}
	return "narrative"
}

func splitByRunes(text string, chunkSize int, overlap int) []string {
	runes := []rune(strings.TrimSpace(text))
	if len(runes) == 0 || chunkSize <= 0 {
		return nil
	}
	if overlap >= chunkSize {
		overlap = 0
	}
	step := chunkSize - overlap
	out := make([]string, 0, (len(runes)/step)+1)
	for start := 0; start < len(runes); start += step {
		end := start + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			out = append(out, chunk)
		}
		if end == len(runes) {
			break
		}
	}
	return out
}

func limitRunes(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max])
}

func hashID(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:8])
}
