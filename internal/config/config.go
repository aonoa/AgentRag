package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	VectorBackendMemory = "memory"
	VectorBackendQdrant = "qdrant"

	SQLDriverSQLite    = "sqlite"
	SQLDriverPostgres  = "pgx"
	SQLDriverMySQL     = "mysql"
	SQLDriverSQLServer = "sqlserver"
)

type Config struct {
	ServerAddr string
	LogLevel   string

	UseOpenAI              bool
	OpenAILLMBaseURL       string
	OpenAIEmbeddingBaseURL string
	OpenAILLMAPIKey        string
	OpenAIEmbeddingAPIKey  string
	OpenAIAPIKey           string
	LLMModel               string
	EmbeddingModel         string
	EmbeddingDim           int
	HTTPTimeout            time.Duration
	QdrantTimeout          time.Duration
	VectorBackend          string
	QdrantURL              string
	QdrantHost             string
	QdrantGRPCPort         int
	QdrantAPIKey           string
	QdrantUseTLS           bool

	ChunkCollection   string
	SummaryCollection string

	ChunkSize    int
	ChunkOverlap int
	TopK         int
	MaxRetries   int
	RerankTopM   int

	RerankURL   string
	RerankKey   string
	RerankModel string

	RouterModel string
	GradeModel  string

	SQLDSN    string
	SQLDriver string

	SerperAPIKey string
	SkillsDir    string

	PlannerMaxSubqueries       int
	OrchestratorMaxExternal    int
	OrchestratorTimeoutSeconds int
	EarlyStopMinCandidates     int
	EarlyStopTopScore          float64
	DirectConfidenceThreshold  float64
	DirectAutoFallback         bool
	DirectFallbackRoute        string
}

func Load() (Config, error) {
	cfg := builtinDefaults()

	fileKV, err := loadConfigFileKV()
	if err != nil {
		return Config{}, err
	}
	if err := applyKV(&cfg, fileKV); err != nil {
		return Config{}, err
	}

	envKV := currentEnvKV()
	if err := applyKV(&cfg, envKV); err != nil {
		return Config{}, err
	}

	if err := validate(cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func builtinDefaults() Config {
	return Config{
		ServerAddr:                 ":8080",
		LogLevel:                   "info",
		UseOpenAI:                  false,
		OpenAILLMBaseURL:           "https://api.openai.com",
		OpenAIEmbeddingBaseURL:     "https://api.openai.com",
		OpenAILLMAPIKey:            "",
		OpenAIEmbeddingAPIKey:      "",
		OpenAIAPIKey:               "",
		LLMModel:                   "gpt-5.4-mini",
		EmbeddingModel:             "text-embedding-3-small",
		EmbeddingDim:               1536,
		HTTPTimeout:                30 * time.Second,
		QdrantTimeout:              20 * time.Second,
		VectorBackend:              VectorBackendMemory,
		QdrantURL:                  "http://localhost:6333",
		QdrantHost:                 "localhost",
		QdrantGRPCPort:             6334,
		QdrantAPIKey:               "",
		QdrantUseTLS:               false,
		ChunkCollection:            "rag_chunks",
		SummaryCollection:          "rag_summaries",
		ChunkSize:                  800,
		ChunkOverlap:               120,
		TopK:                       6,
		MaxRetries:                 3,
		RerankTopM:                 4,
		RerankURL:                  "",
		RerankKey:                  "",
		RerankModel:                "BAAI/bge-reranker-v2-m3",
		RouterModel:                "gpt-5.4-mini",
		GradeModel:                 "gpt-5.4-mini",
		SQLDSN:                     "",
		SQLDriver:                  SQLDriverSQLite,
		SerperAPIKey:               "",
		SkillsDir:                  "",
		PlannerMaxSubqueries:       3,
		OrchestratorMaxExternal:    2,
		OrchestratorTimeoutSeconds: 8,
		EarlyStopMinCandidates:     3,
		EarlyStopTopScore:          0.045,
		DirectConfidenceThreshold:  0.70,
		DirectAutoFallback:         true,
		DirectFallbackRoute:        "catalog",
	}
}

func loadConfigFileKV() (map[string]any, error) {
	if p, ok := os.LookupEnv("CONFIG_FILE"); ok {
		trim := strings.TrimSpace(p)
		if trim != "" {
			baseRaw, err := readJSONFile(trim, true)
			if err != nil {
				return nil, err
			}
			merged := toCanonicalKV(baseRaw)
			envName, explicitEnv := profileName()
			overlayPath := filepath.Join(filepath.Dir(trim), "environments", envName+".json")
			overlayRaw, err := readJSONFile(overlayPath, explicitEnv)
			if err != nil {
				return nil, err
			}
			overlayKV := toCanonicalKV(overlayRaw)
			for k, v := range overlayKV {
				merged[k] = v
			}
			return merged, nil
		}
	}

	baseRaw, err := readJSONFile("config/defaults.json", false)
	if err != nil {
		return nil, err
	}
	merged := toCanonicalKV(baseRaw)

	envName, explicitEnv := profileName()
	overlayPath := filepath.Join("config", "environments", envName+".json")
	overlayRaw, err := readJSONFile(overlayPath, explicitEnv)
	if err != nil {
		return nil, err
	}
	overlayKV := toCanonicalKV(overlayRaw)
	for k, v := range overlayKV {
		merged[k] = v
	}

	return merged, nil
}

func profileName() (string, bool) {
	if v, ok := os.LookupEnv("APP_ENV"); ok {
		trim := strings.ToLower(strings.TrimSpace(v))
		if trim != "" {
			return trim, true
		}
	}
	return "default", false
}

func readJSONFile(path string, required bool) (map[string]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) && !required {
			return map[string]any{}, nil
		}
		return nil, fmt.Errorf("read config file %s: %w", path, err)
	}
	m := make(map[string]any)
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("parse config file %s: %w", path, err)
	}
	return m, nil
}

func toCanonicalKV(raw map[string]any) map[string]any {
	out := make(map[string]any)
	for _, key := range []string{
		"SERVER_ADDR", "LOG_LEVEL", "USE_OPENAI", "OPENAI_BASE_URL", "OPENAI_LLM_BASE_URL", "OPENAI_EMBEDDING_BASE_URL", "OPENAI_API_KEY", "OPENAI_LLM_API_KEY", "OPENAI_EMBEDDING_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL",
		"EMBEDDING_DIM", "HTTP_TIMEOUT_SECONDS", "QDRANT_TIMEOUT_SECONDS", "VECTOR_BACKEND", "QDRANT_URL", "QDRANT_HOST",
		"QDRANT_GRPC_PORT", "QDRANT_API_KEY", "QDRANT_USE_TLS", "QDRANT_CHUNK_COLLECTION", "QDRANT_SUMMARY_COLLECTION",
		"CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K", "MAX_RETRY_LOOPS", "RERANK_TOP_M", "RERANK_URL",
		"RERANK_API_KEY", "RERANK_MODEL", "ROUTER_MODEL", "GRADE_MODEL", "SQL_DSN", "SQL_DRIVER", "SERPER_API_KEY", "SKILLS_DIR",
		"PLANNER_MAX_SUBQUERIES", "ORCHESTRATOR_MAX_EXTERNAL_CALLS", "ORCHESTRATOR_TIMEOUT_SECONDS", "EARLY_STOP_MIN_CANDIDATES", "EARLY_STOP_TOP_SCORE",
		"DIRECT_CONFIDENCE_THRESHOLD", "DIRECT_AUTO_FALLBACK", "DIRECT_FALLBACK_ROUTE",
	} {
		if v, ok := raw[key]; ok {
			out[key] = v
		}
	}

	assignNested(raw, out, "SERVER_ADDR", "app", "server_addr")
	assignNested(raw, out, "LOG_LEVEL", "app", "log_level")

	assignNested(raw, out, "USE_OPENAI", "llm", "use_openai")
	assignNested(raw, out, "OPENAI_BASE_URL", "llm", "openai_base_url")
	assignNested(raw, out, "OPENAI_LLM_BASE_URL", "llm", "openai_llm_base_url")
	assignNested(raw, out, "OPENAI_EMBEDDING_BASE_URL", "llm", "openai_embedding_base_url")
	assignNested(raw, out, "OPENAI_API_KEY", "llm", "openai_api_key")
	assignNested(raw, out, "OPENAI_LLM_API_KEY", "llm", "openai_llm_api_key")
	assignNested(raw, out, "OPENAI_LLM_API_KEY", "llm", "openai_api_key")
	assignNested(raw, out, "LLM_MODEL", "llm", "model")
	assignNested(raw, out, "EMBEDDING_MODEL", "llm", "embedding_model")
	assignNested(raw, out, "EMBEDDING_DIM", "llm", "embedding_dim")

	assignNested(raw, out, "OPENAI_EMBEDDING_BASE_URL", "embedding", "openai_base_url")
	assignNested(raw, out, "OPENAI_EMBEDDING_BASE_URL", "embedding", "openai_embedding_base_url")
	assignNested(raw, out, "OPENAI_EMBEDDING_API_KEY", "embedding", "openai_embedding_api_key")
	assignNested(raw, out, "OPENAI_EMBEDDING_API_KEY", "embedding", "openai_api_key")
	assignNested(raw, out, "EMBEDDING_MODEL", "embedding", "model")
	assignNested(raw, out, "EMBEDDING_MODEL", "embedding", "embedding_model")
	assignNested(raw, out, "EMBEDDING_DIM", "embedding", "dim")
	assignNested(raw, out, "EMBEDDING_DIM", "embedding", "embedding_dim")

	assignNested(raw, out, "HTTP_TIMEOUT_SECONDS", "timeouts", "http_seconds")
	assignNested(raw, out, "QDRANT_TIMEOUT_SECONDS", "timeouts", "qdrant_seconds")

	assignNested(raw, out, "VECTOR_BACKEND", "vector", "backend")
	assignNested(raw, out, "QDRANT_URL", "vector", "qdrant", "url")
	assignNested(raw, out, "QDRANT_HOST", "vector", "qdrant", "host")
	assignNested(raw, out, "QDRANT_GRPC_PORT", "vector", "qdrant", "grpc_port")
	assignNested(raw, out, "QDRANT_API_KEY", "vector", "qdrant", "api_key")
	assignNested(raw, out, "QDRANT_USE_TLS", "vector", "qdrant", "use_tls")
	assignNested(raw, out, "QDRANT_CHUNK_COLLECTION", "vector", "collections", "chunk")
	assignNested(raw, out, "QDRANT_SUMMARY_COLLECTION", "vector", "collections", "summary")

	assignNested(raw, out, "CHUNK_SIZE", "retrieval", "chunk_size")
	assignNested(raw, out, "CHUNK_OVERLAP", "retrieval", "chunk_overlap")
	assignNested(raw, out, "RETRIEVAL_TOP_K", "retrieval", "top_k")
	assignNested(raw, out, "MAX_RETRY_LOOPS", "retrieval", "max_retries")

	assignNested(raw, out, "RERANK_TOP_M", "rerank", "top_m")
	assignNested(raw, out, "RERANK_URL", "rerank", "url")
	assignNested(raw, out, "RERANK_API_KEY", "rerank", "api_key")
	assignNested(raw, out, "RERANK_MODEL", "rerank", "model")

	assignNested(raw, out, "ROUTER_MODEL", "routing", "router_model")
	assignNested(raw, out, "GRADE_MODEL", "routing", "grade_model")

	assignNested(raw, out, "SQL_DRIVER", "sql", "driver")
	assignNested(raw, out, "SQL_DSN", "sql", "dsn")

	assignNested(raw, out, "SERPER_API_KEY", "web", "serper_api_key")
	assignNested(raw, out, "SKILLS_DIR", "skills", "base_dir")

	assignNested(raw, out, "PLANNER_MAX_SUBQUERIES", "orchestration", "planner_max_subqueries")
	assignNested(raw, out, "ORCHESTRATOR_MAX_EXTERNAL_CALLS", "orchestration", "max_external_calls")
	assignNested(raw, out, "ORCHESTRATOR_TIMEOUT_SECONDS", "orchestration", "timeout_seconds")
	assignNested(raw, out, "EARLY_STOP_MIN_CANDIDATES", "orchestration", "early_stop_min_candidates")
	assignNested(raw, out, "EARLY_STOP_TOP_SCORE", "orchestration", "early_stop_top_score")
	assignNested(raw, out, "DIRECT_CONFIDENCE_THRESHOLD", "orchestration", "direct_confidence_threshold")
	assignNested(raw, out, "DIRECT_AUTO_FALLBACK", "orchestration", "direct_auto_fallback")
	assignNested(raw, out, "DIRECT_FALLBACK_ROUTE", "orchestration", "direct_fallback_route")
	return out
}

func assignNested(src map[string]any, out map[string]any, target string, path ...string) {
	v, ok := nestedValue(src, path...)
	if !ok {
		return
	}
	out[target] = v
}

func nestedValue(src map[string]any, path ...string) (any, bool) {
	var cur any = src
	for _, p := range path {
		obj, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		v, ok := obj[p]
		if !ok {
			return nil, false
		}
		cur = v
	}
	return cur, true
}

func currentEnvKV() map[string]any {
	kv := make(map[string]any)
	for _, key := range []string{
		"SERVER_ADDR", "LOG_LEVEL", "USE_OPENAI", "OPENAI_BASE_URL", "OPENAI_LLM_BASE_URL", "OPENAI_EMBEDDING_BASE_URL", "OPENAI_API_KEY", "OPENAI_LLM_API_KEY", "OPENAI_EMBEDDING_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL",
		"EMBEDDING_DIM", "HTTP_TIMEOUT_SECONDS", "QDRANT_TIMEOUT_SECONDS", "VECTOR_BACKEND", "QDRANT_URL",
		"QDRANT_HOST", "QDRANT_GRPC_PORT", "QDRANT_API_KEY", "QDRANT_USE_TLS", "QDRANT_CHUNK_COLLECTION",
		"QDRANT_SUMMARY_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K", "MAX_RETRY_LOOPS",
		"RERANK_TOP_M", "RERANK_URL", "RERANK_API_KEY", "RERANK_MODEL", "ROUTER_MODEL", "GRADE_MODEL",
		"SQL_DSN", "SQL_DRIVER", "SERPER_API_KEY", "SKILLS_DIR",
		"PLANNER_MAX_SUBQUERIES", "ORCHESTRATOR_MAX_EXTERNAL_CALLS", "ORCHESTRATOR_TIMEOUT_SECONDS", "EARLY_STOP_MIN_CANDIDATES", "EARLY_STOP_TOP_SCORE",
		"DIRECT_CONFIDENCE_THRESHOLD", "DIRECT_AUTO_FALLBACK", "DIRECT_FALLBACK_ROUTE",
	} {
		if val, ok := os.LookupEnv(key); ok {
			trimmed := strings.TrimSpace(val)
			if trimmed == "" {
				continue
			}
			kv[key] = trimmed
		}
	}
	return kv
}

func applyKV(cfg *Config, kv map[string]any) error {
	if v, ok, err := kvString(kv, "SERVER_ADDR"); err != nil {
		return err
	} else if ok {
		cfg.ServerAddr = v
	}
	if v, ok, err := kvString(kv, "LOG_LEVEL"); err != nil {
		return err
	} else if ok {
		cfg.LogLevel = strings.ToLower(v)
	}
	if v, ok, err := kvBool(kv, "USE_OPENAI"); err != nil {
		return err
	} else if ok {
		cfg.UseOpenAI = v
	}
	if v, ok, err := kvString(kv, "OPENAI_BASE_URL"); err != nil {
		return err
	} else if ok {
		base := strings.TrimRight(v, "/")
		cfg.OpenAILLMBaseURL = base
		cfg.OpenAIEmbeddingBaseURL = base
	}
	if v, ok, err := kvString(kv, "OPENAI_LLM_BASE_URL"); err != nil {
		return err
	} else if ok {
		cfg.OpenAILLMBaseURL = strings.TrimRight(v, "/")
	}
	if v, ok, err := kvString(kv, "OPENAI_EMBEDDING_BASE_URL"); err != nil {
		return err
	} else if ok {
		cfg.OpenAIEmbeddingBaseURL = strings.TrimRight(v, "/")
	}
	if v, ok, err := kvString(kv, "OPENAI_API_KEY"); err != nil {
		return err
	} else if ok {
		cfg.OpenAIAPIKey = strings.TrimSpace(v)
		cfg.OpenAILLMAPIKey = strings.TrimSpace(v)
		cfg.OpenAIEmbeddingAPIKey = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "OPENAI_LLM_API_KEY"); err != nil {
		return err
	} else if ok {
		cfg.OpenAILLMAPIKey = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "OPENAI_EMBEDDING_API_KEY"); err != nil {
		return err
	} else if ok {
		cfg.OpenAIEmbeddingAPIKey = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "LLM_MODEL"); err != nil {
		return err
	} else if ok {
		cfg.LLMModel = v
	}
	if v, ok, err := kvString(kv, "EMBEDDING_MODEL"); err != nil {
		return err
	} else if ok {
		cfg.EmbeddingModel = v
	}
	if v, ok, err := kvInt(kv, "EMBEDDING_DIM"); err != nil {
		return err
	} else if ok {
		cfg.EmbeddingDim = v
	}
	if v, ok, err := kvInt(kv, "HTTP_TIMEOUT_SECONDS"); err != nil {
		return err
	} else if ok {
		cfg.HTTPTimeout = time.Duration(v) * time.Second
	}
	if v, ok, err := kvInt(kv, "QDRANT_TIMEOUT_SECONDS"); err != nil {
		return err
	} else if ok {
		cfg.QdrantTimeout = time.Duration(v) * time.Second
	}
	if v, ok, err := kvString(kv, "VECTOR_BACKEND"); err != nil {
		return err
	} else if ok {
		cfg.VectorBackend = strings.ToLower(v)
	}
	if v, ok, err := kvString(kv, "QDRANT_URL"); err != nil {
		return err
	} else if ok {
		cfg.QdrantURL = strings.TrimRight(v, "/")
	}
	if v, ok, err := kvString(kv, "QDRANT_HOST"); err != nil {
		return err
	} else if ok {
		cfg.QdrantHost = v
	}
	if v, ok, err := kvInt(kv, "QDRANT_GRPC_PORT"); err != nil {
		return err
	} else if ok {
		cfg.QdrantGRPCPort = v
	}
	if v, ok, err := kvString(kv, "QDRANT_API_KEY"); err != nil {
		return err
	} else if ok {
		cfg.QdrantAPIKey = strings.TrimSpace(v)
	}
	if v, ok, err := kvBool(kv, "QDRANT_USE_TLS"); err != nil {
		return err
	} else if ok {
		cfg.QdrantUseTLS = v
	}
	if v, ok, err := kvString(kv, "QDRANT_CHUNK_COLLECTION"); err != nil {
		return err
	} else if ok {
		cfg.ChunkCollection = v
	}
	if v, ok, err := kvString(kv, "QDRANT_SUMMARY_COLLECTION"); err != nil {
		return err
	} else if ok {
		cfg.SummaryCollection = v
	}
	if v, ok, err := kvInt(kv, "CHUNK_SIZE"); err != nil {
		return err
	} else if ok {
		cfg.ChunkSize = v
	}
	if v, ok, err := kvInt(kv, "CHUNK_OVERLAP"); err != nil {
		return err
	} else if ok {
		cfg.ChunkOverlap = v
	}
	if v, ok, err := kvInt(kv, "RETRIEVAL_TOP_K"); err != nil {
		return err
	} else if ok {
		cfg.TopK = v
	}
	if v, ok, err := kvInt(kv, "MAX_RETRY_LOOPS"); err != nil {
		return err
	} else if ok {
		cfg.MaxRetries = v
	}
	if v, ok, err := kvInt(kv, "RERANK_TOP_M"); err != nil {
		return err
	} else if ok {
		cfg.RerankTopM = v
	}
	if v, ok, err := kvString(kv, "RERANK_URL"); err != nil {
		return err
	} else if ok {
		cfg.RerankURL = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "RERANK_API_KEY"); err != nil {
		return err
	} else if ok {
		cfg.RerankKey = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "RERANK_MODEL"); err != nil {
		return err
	} else if ok {
		cfg.RerankModel = v
	}
	if v, ok, err := kvString(kv, "ROUTER_MODEL"); err != nil {
		return err
	} else if ok {
		cfg.RouterModel = v
	}
	if v, ok, err := kvString(kv, "GRADE_MODEL"); err != nil {
		return err
	} else if ok {
		cfg.GradeModel = v
	}
	if v, ok, err := kvString(kv, "SQL_DSN"); err != nil {
		return err
	} else if ok {
		cfg.SQLDSN = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "SQL_DRIVER"); err != nil {
		return err
	} else if ok {
		cfg.SQLDriver = strings.ToLower(v)
	}
	if v, ok, err := kvString(kv, "SERPER_API_KEY"); err != nil {
		return err
	} else if ok {
		cfg.SerperAPIKey = strings.TrimSpace(v)
	}
	if v, ok, err := kvString(kv, "SKILLS_DIR"); err != nil {
		return err
	} else if ok {
		cfg.SkillsDir = strings.TrimSpace(v)
	}
	if v, ok, err := kvInt(kv, "PLANNER_MAX_SUBQUERIES"); err != nil {
		return err
	} else if ok {
		cfg.PlannerMaxSubqueries = v
	}
	if v, ok, err := kvInt(kv, "ORCHESTRATOR_MAX_EXTERNAL_CALLS"); err != nil {
		return err
	} else if ok {
		cfg.OrchestratorMaxExternal = v
	}
	if v, ok, err := kvInt(kv, "ORCHESTRATOR_TIMEOUT_SECONDS"); err != nil {
		return err
	} else if ok {
		cfg.OrchestratorTimeoutSeconds = v
	}
	if v, ok, err := kvInt(kv, "EARLY_STOP_MIN_CANDIDATES"); err != nil {
		return err
	} else if ok {
		cfg.EarlyStopMinCandidates = v
	}
	if v, ok, err := kvFloat(kv, "EARLY_STOP_TOP_SCORE"); err != nil {
		return err
	} else if ok {
		cfg.EarlyStopTopScore = v
	}
	if v, ok, err := kvFloat(kv, "DIRECT_CONFIDENCE_THRESHOLD"); err != nil {
		return err
	} else if ok {
		cfg.DirectConfidenceThreshold = v
	}
	if v, ok, err := kvBool(kv, "DIRECT_AUTO_FALLBACK"); err != nil {
		return err
	} else if ok {
		cfg.DirectAutoFallback = v
	}
	if v, ok, err := kvString(kv, "DIRECT_FALLBACK_ROUTE"); err != nil {
		return err
	} else if ok {
		cfg.DirectFallbackRoute = strings.ToLower(strings.TrimSpace(v))
	}

	return nil
}

func validate(cfg Config) error {
	if cfg.VectorBackend != VectorBackendMemory && cfg.VectorBackend != VectorBackendQdrant {
		return fmt.Errorf("VECTOR_BACKEND must be one of [%s,%s]", VectorBackendMemory, VectorBackendQdrant)
	}
	if cfg.ChunkSize < 100 {
		return errors.New("CHUNK_SIZE must be >= 100")
	}
	if cfg.ChunkOverlap < 0 || cfg.ChunkOverlap >= cfg.ChunkSize {
		return errors.New("CHUNK_OVERLAP must be >=0 and < CHUNK_SIZE")
	}
	if cfg.TopK <= 0 {
		return errors.New("RETRIEVAL_TOP_K must be > 0")
	}
	if cfg.MaxRetries <= 0 {
		return errors.New("MAX_RETRY_LOOPS must be > 0")
	}
	if cfg.RerankTopM <= 0 {
		return errors.New("RERANK_TOP_M must be > 0")
	}
	if cfg.EmbeddingDim <= 0 {
		return errors.New("EMBEDDING_DIM must be > 0")
	}
	if cfg.UseOpenAI && cfg.OpenAILLMAPIKey == "" {
		return errors.New("OPENAI_LLM_API_KEY (or OPENAI_API_KEY fallback) is required when USE_OPENAI=true")
	}
	if cfg.UseOpenAI && cfg.OpenAIEmbeddingAPIKey == "" {
		return errors.New("OPENAI_EMBEDDING_API_KEY (or OPENAI_API_KEY fallback) is required when USE_OPENAI=true")
	}
	if cfg.VectorBackend == VectorBackendQdrant {
		if cfg.QdrantHost == "" {
			return errors.New("QDRANT_HOST is required when VECTOR_BACKEND=qdrant")
		}
		if cfg.QdrantGRPCPort <= 0 {
			return errors.New("QDRANT_GRPC_PORT must be > 0 when VECTOR_BACKEND=qdrant")
		}
	}
	if cfg.SQLDSN != "" {
		supported := map[string]bool{
			SQLDriverSQLite:    true,
			SQLDriverPostgres:  true,
			SQLDriverMySQL:     true,
			SQLDriverSQLServer: true,
		}
		if !supported[cfg.SQLDriver] {
			return fmt.Errorf("SQL_DRIVER must be one of [%s,%s,%s,%s]", SQLDriverSQLite, SQLDriverPostgres, SQLDriverMySQL, SQLDriverSQLServer)
		}
	}
	if cfg.PlannerMaxSubqueries <= 0 {
		return errors.New("PLANNER_MAX_SUBQUERIES must be > 0")
	}
	if cfg.OrchestratorMaxExternal < 0 {
		return errors.New("ORCHESTRATOR_MAX_EXTERNAL_CALLS must be >= 0")
	}
	if cfg.OrchestratorTimeoutSeconds <= 0 {
		return errors.New("ORCHESTRATOR_TIMEOUT_SECONDS must be > 0")
	}
	if cfg.EarlyStopMinCandidates < 0 {
		return errors.New("EARLY_STOP_MIN_CANDIDATES must be >= 0")
	}
	if cfg.EarlyStopTopScore < 0 {
		return errors.New("EARLY_STOP_TOP_SCORE must be >= 0")
	}
	if cfg.DirectConfidenceThreshold < 0 || cfg.DirectConfidenceThreshold > 1 {
		return errors.New("DIRECT_CONFIDENCE_THRESHOLD must be between 0 and 1")
	}
	if cfg.DirectFallbackRoute != "" {
		validRoutes := map[string]bool{
			"direct": true, "direct_chunk": true, "catalog": true,
			"hierarchical": true, "sql": true, "web": true, "hybrid": true,
		}
		if !validRoutes[cfg.DirectFallbackRoute] {
			return errors.New("DIRECT_FALLBACK_ROUTE must be one of [direct,direct_chunk,catalog,hierarchical,sql,web,hybrid]")
		}
	}
	return nil
}

func kvString(kv map[string]any, key string) (string, bool, error) {
	v, ok := kv[key]
	if !ok {
		return "", false, nil
	}
	switch tv := v.(type) {
	case string:
		return strings.TrimSpace(tv), true, nil
	case float64:
		if math.Mod(tv, 1) == 0 {
			return strconv.Itoa(int(tv)), true, nil
		}
		return strconv.FormatFloat(tv, 'f', -1, 64), true, nil
	case bool:
		if tv {
			return "true", true, nil
		}
		return "false", true, nil
	default:
		return "", false, fmt.Errorf("invalid string value for %s", key)
	}
}

func kvInt(kv map[string]any, key string) (int, bool, error) {
	v, ok := kv[key]
	if !ok {
		return 0, false, nil
	}
	switch tv := v.(type) {
	case float64:
		if math.Mod(tv, 1) != 0 {
			return 0, false, fmt.Errorf("invalid integer value for %s", key)
		}
		return int(tv), true, nil
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(tv))
		if err != nil {
			return 0, false, fmt.Errorf("invalid integer value for %s", key)
		}
		return n, true, nil
	default:
		return 0, false, fmt.Errorf("invalid integer value for %s", key)
	}
}

func kvFloat(kv map[string]any, key string) (float64, bool, error) {
	v, ok := kv[key]
	if !ok {
		return 0, false, nil
	}
	switch tv := v.(type) {
	case float64:
		return tv, true, nil
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(tv), 64)
		if err != nil {
			return 0, false, fmt.Errorf("invalid float value for %s", key)
		}
		return f, true, nil
	default:
		return 0, false, fmt.Errorf("invalid float value for %s", key)
	}
}

func kvBool(kv map[string]any, key string) (bool, bool, error) {
	v, ok := kv[key]
	if !ok {
		return false, false, nil
	}
	switch tv := v.(type) {
	case bool:
		return tv, true, nil
	case string:
		b, err := strconv.ParseBool(strings.TrimSpace(strings.ToLower(tv)))
		if err != nil {
			return false, false, fmt.Errorf("invalid boolean value for %s", key)
		}
		return b, true, nil
	case float64:
		if tv == 1 {
			return true, true, nil
		}
		if tv == 0 {
			return false, true, nil
		}
		return false, false, fmt.Errorf("invalid boolean value for %s", key)
	default:
		return false, false, fmt.Errorf("invalid boolean value for %s", key)
	}
}
