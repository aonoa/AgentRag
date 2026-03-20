package store

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"agentragplus/internal/domain"

	"github.com/google/uuid"
	qdrant "github.com/qdrant/go-client/qdrant"
)

type QdrantStore struct {
	client *qdrant.Client
	dim    int
}

func NewQdrantStore(host string, grpcPort int, apiKey string, useTLS bool, embeddingDim int) (*QdrantStore, error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   grpcPort,
		APIKey: apiKey,
		UseTLS: useTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("create qdrant sdk client: %w", err)
	}
	return &QdrantStore{client: client, dim: embeddingDim}, nil
}

func (q *QdrantStore) EnsureCollections(ctx context.Context, chunkCollection string, summaryCollection string) error {
	for _, col := range []string{chunkCollection, summaryCollection} {
		exists, err := q.client.CollectionExists(ctx, col)
		if err != nil {
			return fmt.Errorf("check qdrant collection %s: %w", col, err)
		}
		if exists {
			continue
		}
		if err := q.client.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: col,
			VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
				"dense": {
					Size:     uint64(q.dim),
					Distance: qdrant.Distance_Cosine,
				},
			}),
			SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
				"sparse": {},
			}),
		}); err != nil {
			return fmt.Errorf("create qdrant collection %s: %w", col, err)
		}
	}
	return nil
}

func (q *QdrantStore) UpsertChunks(ctx context.Context, collection string, chunks []domain.Chunk) error {
	points := make([]*qdrant.PointStruct, 0, len(chunks))
	for _, c := range chunks {
		sparseIdx, sparseVals := sparseTermVector(c.Text)
		payload := map[string]any{
			"chunk_id":      c.ChunkID,
			"document_id":   c.DocumentID,
			"summary_id":    c.SummaryID,
			"source":        c.Source,
			"text":          c.Text,
			"chunk_index":   c.ChunkIndex,
			"chunk_count":   c.ChunkCount,
			"content_class": c.ContentClass,
		}
		points = append(points, &qdrant.PointStruct{
			Id: q.pointIDForExternalID("chunk", c.ChunkID),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"dense":  qdrant.NewVectorDense(toFloat32Vec(c.Embedding)),
				"sparse": qdrant.NewVectorSparse(sparseIdx, sparseVals),
			}),
			Payload: qdrant.NewValueMap(payload),
		})
	}
	if len(points) == 0 {
		return nil
	}
	wait := true
	if _, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: collection, Points: points, Wait: &wait}); err != nil {
		if !isQdrantSchemaCompatibilityError(err) {
			return fmt.Errorf("qdrant upsert chunks: %w", err)
		}
		fallback := make([]*qdrant.PointStruct, 0, len(chunks))
		for _, c := range chunks {
			payload := map[string]any{
				"chunk_id":      c.ChunkID,
				"document_id":   c.DocumentID,
				"summary_id":    c.SummaryID,
				"source":        c.Source,
				"text":          c.Text,
				"chunk_index":   c.ChunkIndex,
				"chunk_count":   c.ChunkCount,
				"content_class": c.ContentClass,
			}
			fallback = append(fallback, &qdrant.PointStruct{Id: q.pointIDForExternalID("chunk", c.ChunkID), Vectors: qdrant.NewVectors(toFloat32Vec(c.Embedding)...), Payload: qdrant.NewValueMap(payload)})
		}
		if _, e2 := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: collection, Points: fallback, Wait: &wait}); e2 != nil {
			return fmt.Errorf("qdrant upsert chunks fallback failed: %w", e2)
		}
	}
	return nil
}

func (q *QdrantStore) UpsertSummaries(ctx context.Context, collection string, summaries []domain.Summary) error {
	points := make([]*qdrant.PointStruct, 0, len(summaries))
	for _, s := range summaries {
		sparseIdx, sparseVals := sparseTermVector(s.Text)
		payload := map[string]any{
			"chunk_id":      s.SummaryID,
			"summary_id":    s.SummaryID,
			"document_id":   s.DocumentID,
			"source":        s.Source,
			"text":          s.Text,
			"content_class": s.ContentClass,
		}
		points = append(points, &qdrant.PointStruct{
			Id: q.pointIDForExternalID("summary", s.SummaryID),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"dense":  qdrant.NewVectorDense(toFloat32Vec(s.Embedding)),
				"sparse": qdrant.NewVectorSparse(sparseIdx, sparseVals),
			}),
			Payload: qdrant.NewValueMap(payload),
		})
	}
	if len(points) == 0 {
		return nil
	}
	wait := true
	if _, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: collection, Points: points, Wait: &wait}); err != nil {
		if !isQdrantSchemaCompatibilityError(err) {
			return fmt.Errorf("qdrant upsert summaries: %w", err)
		}
		fallback := make([]*qdrant.PointStruct, 0, len(summaries))
		for _, s := range summaries {
			payload := map[string]any{
				"chunk_id":      s.SummaryID,
				"summary_id":    s.SummaryID,
				"document_id":   s.DocumentID,
				"source":        s.Source,
				"text":          s.Text,
				"content_class": s.ContentClass,
			}
			fallback = append(fallback, &qdrant.PointStruct{Id: q.pointIDForExternalID("summary", s.SummaryID), Vectors: qdrant.NewVectors(toFloat32Vec(s.Embedding)...), Payload: qdrant.NewValueMap(payload)})
		}
		if _, e2 := q.client.Upsert(ctx, &qdrant.UpsertPoints{CollectionName: collection, Points: fallback, Wait: &wait}); e2 != nil {
			return fmt.Errorf("qdrant upsert summaries fallback failed: %w", e2)
		}
	}
	return nil
}

func (q *QdrantStore) SearchChunks(ctx context.Context, collection string, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	points, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(toFloat32Vec(embedding)...),
		Using:          strPtr("dense"),
		Limit:          uint64Ptr(uint64(topK)),
		Filter:         q.buildFilter(filter),
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		if !isQdrantSchemaCompatibilityError(err) {
			return nil, fmt.Errorf("qdrant query chunks: %w", err)
		}
		points, err = q.client.Query(ctx, &qdrant.QueryPoints{CollectionName: collection, Query: qdrant.NewQuery(toFloat32Vec(embedding)...), Limit: uint64Ptr(uint64(topK)), Filter: q.buildFilter(filter), WithPayload: qdrant.NewWithPayload(true)})
		if err != nil {
			return nil, fmt.Errorf("qdrant query chunks fallback failed: %w", err)
		}
	}
	out := make([]domain.RetrievalCandidate, 0, len(points))
	for _, p := range points {
		row := q.pointToCandidate(p)
		row.Layer = "chunk"
		out = append(out, row)
	}
	return out, nil
}

func (q *QdrantStore) SearchChunksSparse(ctx context.Context, collection string, query string, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	indices, values := sparseTermVector(query)
	if len(indices) == 0 {
		return nil, nil
	}
	points, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuerySparse(indices, values),
		Using:          strPtr("sparse"),
		Limit:          uint64Ptr(uint64(topK)),
		Filter:         q.buildFilter(filter),
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		if !isQdrantSchemaCompatibilityError(err) {
			return nil, fmt.Errorf("qdrant sparse query failed: %w", err)
		}
		rows, e2 := q.client.Scroll(ctx, &qdrant.ScrollPoints{CollectionName: collection, Filter: q.buildFilter(filter), WithPayload: qdrant.NewWithPayload(true), Limit: uint32Ptr(uint32(max(2000, topK*20)))})
		if e2 != nil {
			return nil, fmt.Errorf("qdrant sparse fallback scroll failed: %w", e2)
		}
		out := make([]domain.RetrievalCandidate, 0, len(rows))
		docs := make([]bm25Doc, 0, len(rows))
		candidateByID := make(map[string]domain.RetrievalCandidate, len(rows))
		qtokens := sparseTokens(query)
		for _, row := range rows {
			cand := q.retrievedToCandidate(row)
			cand.Layer = "sparse_chunk"
			docs = append(docs, bm25Doc{ID: cand.ChunkID, Tokens: sparseTokens(cand.Text)})
			candidateByID[cand.ChunkID] = cand
		}
		model := buildBM25Model(docs)
		for _, doc := range docs {
			score := model.score(doc.ID, qtokens)
			if score <= 0 {
				continue
			}
			cand := candidateByID[doc.ID]
			cand.Score = score
			out = append(out, cand)
		}
		sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
		if topK > 0 && len(out) > topK {
			out = out[:topK]
		}
		return out, nil
	}

	out := make([]domain.RetrievalCandidate, 0, len(points))
	for _, p := range points {
		cand := q.pointToCandidate(p)
		cand.Layer = "sparse_chunk"
		out = append(out, cand)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if topK > 0 && len(out) > topK {
		out = out[:topK]
	}
	return out, nil
}

func (q *QdrantStore) SearchSummaries(ctx context.Context, collection string, embedding []float64, topK int, filter map[string]any) ([]domain.RetrievalCandidate, error) {
	points, err := q.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(toFloat32Vec(embedding)...),
		Using:          strPtr("dense"),
		Limit:          uint64Ptr(uint64(topK)),
		Filter:         q.buildFilter(filter),
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		if !isQdrantSchemaCompatibilityError(err) {
			return nil, fmt.Errorf("qdrant query summaries: %w", err)
		}
		points, err = q.client.Query(ctx, &qdrant.QueryPoints{CollectionName: collection, Query: qdrant.NewQuery(toFloat32Vec(embedding)...), Limit: uint64Ptr(uint64(topK)), Filter: q.buildFilter(filter), WithPayload: qdrant.NewWithPayload(true)})
		if err != nil {
			return nil, fmt.Errorf("qdrant query summaries fallback failed: %w", err)
		}
	}
	out := make([]domain.RetrievalCandidate, 0, len(points))
	for _, p := range points {
		row := q.pointToCandidate(p)
		row.Layer = "summary"
		if row.SummaryID == "" {
			row.SummaryID = row.ChunkID
		}
		out = append(out, row)
	}
	return out, nil
}

func (q *QdrantStore) GetChunksBySummaryIDs(ctx context.Context, collection string, summaryIDs []string, topKPerSummary int) ([]domain.RetrievalCandidate, error) {
	if len(summaryIDs) == 0 {
		return nil, nil
	}
	out := make([]domain.RetrievalCandidate, 0)
	for _, sid := range summaryIDs {
		flt := &qdrant.Filter{Must: []*qdrant.Condition{qdrant.NewMatchKeyword("summary_id", sid)}}
		rows, err := q.client.Scroll(ctx, &qdrant.ScrollPoints{
			CollectionName: collection,
			Filter:         flt,
			Limit:          uint32Ptr(uint32(topKPerSummary)),
			WithPayload:    qdrant.NewWithPayload(true),
		})
		if err != nil {
			return nil, fmt.Errorf("qdrant scroll by summary_id: %w", err)
		}
		for _, row := range rows {
			out = append(out, q.retrievedToCandidate(row))
		}
	}
	return out, nil
}

func (q *QdrantStore) buildFilter(filter map[string]any) *qdrant.Filter {
	if len(filter) == 0 {
		return nil
	}
	must := make([]*qdrant.Condition, 0, len(filter))
	for key, value := range filter {
		switch tv := value.(type) {
		case string:
			must = append(must, qdrant.NewMatchKeyword(key, tv))
		case int:
			must = append(must, qdrant.NewMatchInt(key, int64(tv)))
		case int64:
			must = append(must, qdrant.NewMatchInt(key, tv))
		case bool:
			must = append(must, qdrant.NewMatchBool(key, tv))
		}
	}
	if len(must) == 0 {
		return nil
	}
	return &qdrant.Filter{Must: must}
}

func (q *QdrantStore) pointToCandidate(p *qdrant.ScoredPoint) domain.RetrievalCandidate {
	chunkID := payloadString(p.GetPayload(), "chunk_id")
	if chunkID == "" {
		chunkID = pointIDString(p.GetId())
	}
	summaryID := payloadString(p.GetPayload(), "summary_id")
	return domain.RetrievalCandidate{
		ChunkID:      chunkID,
		DocumentID:   payloadString(p.GetPayload(), "document_id"),
		SummaryID:    summaryID,
		Score:        float64(p.GetScore()),
		Source:       payloadString(p.GetPayload(), "source"),
		Text:         payloadString(p.GetPayload(), "text"),
		ContentClass: payloadString(p.GetPayload(), "content_class"),
	}
}

func (q *QdrantStore) retrievedToCandidate(p *qdrant.RetrievedPoint) domain.RetrievalCandidate {
	chunkID := payloadString(p.GetPayload(), "chunk_id")
	if chunkID == "" {
		chunkID = pointIDString(p.GetId())
	}
	summaryID := payloadString(p.GetPayload(), "summary_id")
	return domain.RetrievalCandidate{
		ChunkID:      chunkID,
		DocumentID:   payloadString(p.GetPayload(), "document_id"),
		SummaryID:    summaryID,
		Score:        0,
		Source:       payloadString(p.GetPayload(), "source"),
		Text:         payloadString(p.GetPayload(), "text"),
		ContentClass: payloadString(p.GetPayload(), "content_class"),
		Layer:        "chunk",
	}
}

func (q *QdrantStore) pointIDForExternalID(entityType string, externalID string) *qdrant.PointId {
	name := strings.TrimSpace(entityType) + ":" + strings.TrimSpace(externalID)
	id := uuid.NewSHA1(uuid.NameSpaceOID, []byte(name)).String()
	return qdrant.NewID(id)
}

func pointIDString(id *qdrant.PointId) string {
	if id == nil {
		return ""
	}
	if v := id.GetUuid(); v != "" {
		return v
	}
	if n := id.GetNum(); n != 0 {
		return strconv.FormatUint(n, 10)
	}
	return ""
}

func payloadString(payload map[string]*qdrant.Value, key string) string {
	v, ok := payload[key]
	if !ok || v == nil {
		return ""
	}
	if s := v.GetStringValue(); s != "" {
		return s
	}
	if i := v.GetIntegerValue(); i != 0 {
		return strconv.FormatInt(i, 10)
	}
	if b := v.GetBoolValue(); b {
		return "true"
	}
	if d := v.GetDoubleValue(); d != 0 {
		return strconv.FormatFloat(d, 'f', -1, 64)
	}
	return ""
}

func uint64Ptr(v uint64) *uint64 {
	return &v
}

func uint32Ptr(v uint32) *uint32 {
	return &v
}

func toFloat32Vec(src []float64) []float32 {
	out := make([]float32, 0, len(src))
	for _, v := range src {
		out = append(out, float32(v))
	}
	return out
}

func max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func strPtr(v string) *string {
	return &v
}

func isQdrantSchemaCompatibilityError(err error) bool {
	if err == nil {
		return false
	}
	lower := strings.ToLower(err.Error())
	return strings.Contains(lower, "vector") || strings.Contains(lower, "sparse") || strings.Contains(lower, "not found") || strings.Contains(lower, "wrong input") || strings.Contains(lower, "using")
}
