package retrieval

import (
	"context"
	"fmt"
	"sort"

	"agentragplus/internal/config"
	"agentragplus/internal/domain"
	"agentragplus/internal/llm"
	"agentragplus/internal/obs"
	"agentragplus/internal/store"
)

type Retriever struct {
	cfg   config.Config
	llm   llm.Client
	store store.VectorStore
}

func NewRetriever(cfg config.Config, llmClient llm.Client, vectorStore store.VectorStore) *Retriever {
	return &Retriever{cfg: cfg, llm: llmClient, store: vectorStore}
}

func (r *Retriever) DirectChunk(ctx context.Context, question string, filter map[string]any) (domain.RetrievalResult, error) {
	ctx, span := obs.StartSpan(ctx, "retrieval.direct_chunk")
	defer obs.EndSpan(span, nil)
	obs.EmitEvent(ctx, "retrieval.direct_chunk.start")
	embed, err := r.llm.Embed(ctx, r.cfg.EmbeddingModel, []string{question})
	if err != nil {
		return domain.RetrievalResult{}, fmt.Errorf("embed question: %w", err)
	}
	cands, err := r.store.SearchChunks(ctx, r.cfg.ChunkCollection, embed[0], r.cfg.TopK, filter)
	if err != nil {
		return domain.RetrievalResult{}, err
	}
	return domain.RetrievalResult{
		Candidates: cands,
		Debug: map[string]any{
			"mode":       "direct_chunk",
			"candidates": len(cands),
		},
	}, nil
}

func (r *Retriever) SparseChunk(ctx context.Context, question string, filter map[string]any) (domain.RetrievalResult, error) {
	ctx, span := obs.StartSpan(ctx, "retrieval.sparse_chunk")
	defer obs.EndSpan(span, nil)
	obs.EmitEvent(ctx, "retrieval.sparse_chunk.start")
	cands, err := r.store.SearchChunksSparse(ctx, r.cfg.ChunkCollection, question, r.cfg.TopK, filter)
	if err != nil {
		return domain.RetrievalResult{}, err
	}
	return domain.RetrievalResult{
		Candidates: cands,
		Debug: map[string]any{
			"mode":       "sparse_chunk",
			"candidates": len(cands),
		},
	}, nil
}

func (r *Retriever) Hierarchical(ctx context.Context, question string, filter map[string]any) (domain.RetrievalResult, error) {
	ctx, span := obs.StartSpan(ctx, "retrieval.hierarchical")
	defer obs.EndSpan(span, nil)
	obs.EmitEvent(ctx, "retrieval.hierarchical.start")
	embed, err := r.llm.Embed(ctx, r.cfg.EmbeddingModel, []string{question})
	if err != nil {
		return domain.RetrievalResult{}, fmt.Errorf("embed question: %w", err)
	}
	summaries, err := r.store.SearchSummaries(ctx, r.cfg.SummaryCollection, embed[0], r.cfg.TopK, filter)
	if err != nil {
		return domain.RetrievalResult{}, err
	}
	summaryIDs := make([]string, 0, len(summaries))
	for _, s := range summaries {
		summaryIDs = append(summaryIDs, s.SummaryID)
	}
	chunks, err := r.store.GetChunksBySummaryIDs(ctx, r.cfg.ChunkCollection, summaryIDs, 2)
	if err != nil {
		return domain.RetrievalResult{}, err
	}

	chunkEmbeds, err := r.llm.Embed(ctx, r.cfg.EmbeddingModel, textsOf(chunks))
	if err != nil {
		return domain.RetrievalResult{}, fmt.Errorf("embed chunks for rerank-lite: %w", err)
	}
	qv := embed[0]
	for i := range chunks {
		chunks[i].Score = cosine(qv, chunkEmbeds[i])
		chunks[i].Layer = "hierarchical_chunk"
	}
	sort.Slice(chunks, func(i, j int) bool { return chunks[i].Score > chunks[j].Score })
	if len(chunks) > r.cfg.TopK {
		chunks = chunks[:r.cfg.TopK]
	}

	return domain.RetrievalResult{
		Candidates: chunks,
		Debug: map[string]any{
			"mode":             "hierarchical",
			"summary_hits":     len(summaries),
			"chunk_candidates": len(chunks),
		},
	}, nil
}

func (r *Retriever) SearchSummariesOnly(ctx context.Context, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	rows, err := r.store.SearchSummaries(ctx, r.cfg.SummaryCollection, embedding, topK, filter)
	if err != nil {
		return nil, err
	}
	for i := range rows {
		rows[i].Layer = "catalog_summary"
		if rows[i].SummaryID == "" {
			rows[i].SummaryID = rows[i].ChunkID
		}
	}
	return rows, nil
}

func textsOf(rows []domain.RetrievalCandidate) []string {
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, r.Text)
	}
	return out
}

func cosine(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt(na) * sqrt(nb))
}

func sqrt(x float64) float64 {
	z := x
	if z == 0 {
		return 0
	}
	for i := 0; i < 8; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
