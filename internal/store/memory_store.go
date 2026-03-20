package store

import (
	"context"
	"math"
	"sort"
	"sync"

	"agentragplus/internal/domain"
)

type MemoryStore struct {
	mu        sync.RWMutex
	chunks    map[string][]domain.Chunk
	summaries map[string][]domain.Summary
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		chunks:    make(map[string][]domain.Chunk),
		summaries: make(map[string][]domain.Summary),
	}
}

func (m *MemoryStore) EnsureCollections(_ context.Context, chunkCollection string, summaryCollection string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.chunks[chunkCollection]; !ok {
		m.chunks[chunkCollection] = []domain.Chunk{}
	}
	if _, ok := m.summaries[summaryCollection]; !ok {
		m.summaries[summaryCollection] = []domain.Summary{}
	}
	return nil
}

func (m *MemoryStore) UpsertChunks(_ context.Context, collection string, chunks []domain.Chunk) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	current := m.chunks[collection]
	index := make(map[string]int, len(current))
	for i, c := range current {
		index[c.ChunkID] = i
	}
	for _, c := range chunks {
		if pos, ok := index[c.ChunkID]; ok {
			current[pos] = c
		} else {
			current = append(current, c)
		}
	}
	m.chunks[collection] = current
	return nil
}

func (m *MemoryStore) UpsertSummaries(_ context.Context, collection string, summaries []domain.Summary) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	current := m.summaries[collection]
	index := make(map[string]int, len(current))
	for i, s := range current {
		index[s.SummaryID] = i
	}
	for _, s := range summaries {
		if pos, ok := index[s.SummaryID]; ok {
			current[pos] = s
		} else {
			current = append(current, s)
		}
	}
	m.summaries[collection] = current
	return nil
}

func (m *MemoryStore) SearchChunks(_ context.Context, collection string, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	source := m.chunks[collection]
	rows := make([]domain.RetrievalCandidate, 0, len(source))
	for _, c := range source {
		if !matchFilter(c.DocumentID, filter) {
			continue
		}
		s := cosine(embedding, c.Embedding)
		rows = append(rows, domain.RetrievalCandidate{
			ChunkID:      c.ChunkID,
			DocumentID:   c.DocumentID,
			SummaryID:    c.SummaryID,
			Score:        s,
			Source:       c.Source,
			Text:         c.Text,
			ContentClass: c.ContentClass,
			Layer:        "chunk",
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Score > rows[j].Score })
	if topK > 0 && len(rows) > topK {
		rows = rows[:topK]
	}
	return rows, nil
}

func (m *MemoryStore) SearchChunksSparse(_ context.Context, collection string, query string, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tokens := sparseTokens(query)
	if len(tokens) == 0 {
		return nil, nil
	}

	source := m.chunks[collection]
	docs := make([]bm25Doc, 0, len(source))
	rowByID := make(map[string]domain.RetrievalCandidate, len(source))
	for _, c := range source {
		if !matchFilter(c.DocumentID, filter) {
			continue
		}
		docs = append(docs, bm25Doc{ID: c.ChunkID, Tokens: sparseTokens(c.Text)})
		rowByID[c.ChunkID] = domain.RetrievalCandidate{
			ChunkID:      c.ChunkID,
			DocumentID:   c.DocumentID,
			SummaryID:    c.SummaryID,
			Score:        0,
			Source:       c.Source,
			Text:         c.Text,
			ContentClass: c.ContentClass,
			Layer:        "sparse_chunk",
		}
	}
	model := buildBM25Model(docs)
	rows := make([]domain.RetrievalCandidate, 0, len(source))
	for _, d := range docs {
		score := model.score(d.ID, tokens)
		if score <= 0 {
			continue
		}
		cand := rowByID[d.ID]
		cand.Score = score
		rows = append(rows, cand)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Score > rows[j].Score })
	if topK > 0 && len(rows) > topK {
		rows = rows[:topK]
	}
	return rows, nil
}

func (m *MemoryStore) SearchSummaries(_ context.Context, collection string, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	source := m.summaries[collection]
	rows := make([]domain.RetrievalCandidate, 0, len(source))
	for _, s := range source {
		if !matchFilter(s.DocumentID, filter) {
			continue
		}
		score := cosine(embedding, s.Embedding)
		rows = append(rows, domain.RetrievalCandidate{
			ChunkID:      s.SummaryID,
			DocumentID:   s.DocumentID,
			SummaryID:    s.SummaryID,
			Score:        score,
			Source:       s.Source,
			Text:         s.Text,
			ContentClass: s.ContentClass,
			Layer:        "summary",
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Score > rows[j].Score })
	if topK > 0 && len(rows) > topK {
		rows = rows[:topK]
	}
	return rows, nil
}

func (m *MemoryStore) GetChunksBySummaryIDs(_ context.Context, collection string, summaryIDs []string, topKPerSummary int) ([]domain.RetrievalCandidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	allow := make(map[string]bool, len(summaryIDs))
	for _, id := range summaryIDs {
		allow[id] = true
	}
	groups := make(map[string][]domain.RetrievalCandidate)
	for _, c := range m.chunks[collection] {
		if !allow[c.SummaryID] {
			continue
		}
		groups[c.SummaryID] = append(groups[c.SummaryID], domain.RetrievalCandidate{
			ChunkID:      c.ChunkID,
			DocumentID:   c.DocumentID,
			SummaryID:    c.SummaryID,
			Score:        0,
			Source:       c.Source,
			Text:         c.Text,
			ContentClass: c.ContentClass,
			Layer:        "chunk",
		})
	}
	merged := make([]domain.RetrievalCandidate, 0)
	for _, list := range groups {
		if topKPerSummary > 0 && len(list) > topKPerSummary {
			list = list[:topKPerSummary]
		}
		merged = append(merged, list...)
	}
	return merged, nil
}

func matchFilter(documentID string, filter map[string]any) bool {
	if len(filter) == 0 {
		return true
	}
	wantDocID, ok := filter["document_id"].(string)
	if !ok || wantDocID == "" {
		return true
	}
	return documentID == wantDocID
}

func cosine(a []float64, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot float64
	var na float64
	var nb float64
	for i := 0; i < n; i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
