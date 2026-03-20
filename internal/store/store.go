package store

import (
	"context"

	"agentragplus/internal/domain"
)

// VectorStore defines the backend-agnostic vector adapter contract.
// Implementations include memory, qdrant-sdk, and future vector DBs.
type VectorStore interface {
	EnsureCollections(ctx context.Context, chunkCollection string, summaryCollection string) error
	UpsertChunks(ctx context.Context, collection string, chunks []domain.Chunk) error
	UpsertSummaries(ctx context.Context, collection string, summaries []domain.Summary) error
	SearchChunks(ctx context.Context, collection string, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error)
	SearchChunksSparse(ctx context.Context, collection string, query string, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error)
	SearchSummaries(ctx context.Context, collection string, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error)
	GetChunksBySummaryIDs(ctx context.Context, collection string, summaryIDs []string, topKPerSummary int) ([]domain.RetrievalCandidate, error)
}
