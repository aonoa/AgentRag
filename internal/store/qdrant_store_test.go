package store

import (
	"testing"

	"github.com/google/uuid"
	qdrant "github.com/qdrant/go-client/qdrant"
)

func TestPointIDForExternalIDReturnsNumericID(t *testing.T) {
	qs := &QdrantStore{}
	id := qs.pointIDForExternalID("summary", "sum_103120f68dc22192")
	if id == nil {
		t.Fatal("expected non-nil point id")
	}
	u := id.GetUuid()
	if u == "" {
		t.Fatalf("expected uuid point id, got %+v", id)
	}
	if _, err := uuid.Parse(u); err != nil {
		t.Fatalf("expected valid uuid point id, got %q (%v)", u, err)
	}
}

func TestRetrievedToCandidatePrefersPayloadChunkID(t *testing.T) {
	qs := &QdrantStore{}
	r := &qdrant.RetrievedPoint{
		Id: qdrant.NewIDNum(123),
		Payload: qdrant.NewValueMap(map[string]any{
			"chunk_id":    "sum_103120f68dc22192",
			"summary_id":  "sum_103120f68dc22192",
			"document_id": "doc_1",
			"source":      "unit-test",
			"text":        "hello",
		}),
	}
	c := qs.retrievedToCandidate(r)
	if c.ChunkID != "sum_103120f68dc22192" {
		t.Fatalf("expected chunk_id from payload, got %q", c.ChunkID)
	}
	if c.SummaryID != "sum_103120f68dc22192" {
		t.Fatalf("expected summary_id from payload, got %q", c.SummaryID)
	}
}
