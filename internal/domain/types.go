package domain

import "time"

type RouteType string

const (
	RouteSkill        RouteType = "skill"
	RouteDirect       RouteType = "direct"
	RouteDirectChunk  RouteType = "direct_chunk"
	RouteCatalog      RouteType = "catalog"
	RouteHierarchical RouteType = "hierarchical"
	RouteSQL          RouteType = "sql"
	RouteWeb          RouteType = "web"
	RouteHybrid       RouteType = "hybrid"
)

type AskRequest struct {
	Question     string            `json:"question"`
	SessionID    string            `json:"session_id,omitempty"`
	ForceRoute   RouteType         `json:"force_route,omitempty"`
	Skill        string            `json:"skill,omitempty"`
	SkillArgs    map[string]any    `json:"skill_args,omitempty"`
	Debug        bool              `json:"debug,omitempty"`
	ExtraFilters map[string]string `json:"extra_filters,omitempty"`
}

type AskResponse struct {
	Answer            string         `json:"answer"`
	Route             RouteType      `json:"route"`
	RewrittenQuestion string         `json:"rewritten_question"`
	Attempts          int            `json:"attempts"`
	References        []Reference    `json:"references,omitempty"`
	Debug             map[string]any `json:"debug,omitempty"`
}

type UploadResponse struct {
	DocumentID string `json:"document_id"`
	Filename   string `json:"filename"`
	Chunks     int    `json:"chunks"`
	Summaries  int    `json:"summaries"`
}

type Reference struct {
	DocumentID string  `json:"document_id"`
	ChunkID    string  `json:"chunk_id"`
	Score      float64 `json:"score"`
	Source     string  `json:"source"`
	Content    string  `json:"content"`
}

type Chunk struct {
	ChunkID      string
	DocumentID   string
	SummaryID    string
	Source       string
	Text         string
	Embedding    []float64
	CreatedAt    time.Time
	ChunkIndex   int
	ChunkCount   int
	ContentClass string
}

type Summary struct {
	SummaryID    string
	DocumentID   string
	Source       string
	Text         string
	Embedding    []float64
	CreatedAt    time.Time
	ContentClass string
}

type RetrievalCandidate struct {
	ChunkID      string
	DocumentID   string
	SummaryID    string
	Score        float64
	Source       string
	Text         string
	ContentClass string
	Layer        string
}

type RetrievalResult struct {
	Candidates []RetrievalCandidate
	Debug      map[string]any
}

type GradeResult struct {
	Relevant bool
	Reason   string
	Score    float64
}
