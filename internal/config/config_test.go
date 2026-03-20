package config

import (
	"os"
	"path/filepath"
	"testing"
)

func clearConfigEnv(t *testing.T) {
	t.Helper()
	keys := []string{
		"CONFIG_FILE", "APP_ENV", "SERVER_ADDR", "LOG_LEVEL", "USE_OPENAI", "OPENAI_BASE_URL", "OPENAI_LLM_BASE_URL", "OPENAI_EMBEDDING_BASE_URL", "OPENAI_API_KEY", "OPENAI_LLM_API_KEY", "OPENAI_EMBEDDING_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL",
		"EMBEDDING_DIM", "HTTP_TIMEOUT_SECONDS", "QDRANT_TIMEOUT_SECONDS", "VECTOR_BACKEND", "QDRANT_URL", "QDRANT_HOST", "QDRANT_GRPC_PORT",
		"QDRANT_API_KEY", "QDRANT_USE_TLS", "QDRANT_CHUNK_COLLECTION", "QDRANT_SUMMARY_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP",
		"RETRIEVAL_TOP_K", "MAX_RETRY_LOOPS", "RERANK_TOP_M", "RERANK_URL", "RERANK_API_KEY", "RERANK_MODEL", "ROUTER_MODEL",
		"GRADE_MODEL", "SQL_DSN", "SQL_DRIVER", "SERPER_API_KEY", "SKILLS_DIR", "OBS_ENABLE_TRACING", "OBS_ENABLE_CALLBACKS", "PLANNER_MAX_SUBQUERIES", "ORCHESTRATOR_MAX_EXTERNAL_CALLS", "ORCHESTRATOR_TIMEOUT_SECONDS", "EARLY_STOP_MIN_CANDIDATES", "EARLY_STOP_TOP_SCORE", "DIRECT_CONFIDENCE_THRESHOLD", "DIRECT_AUTO_FALLBACK", "DIRECT_FALLBACK_ROUTE",
	}
	for _, k := range keys {
		t.Setenv(k, "")
	}
}

func TestDirectFallbackConfigFromEnv(t *testing.T) {
	clearConfigEnv(t)
	t.Setenv("DIRECT_CONFIDENCE_THRESHOLD", "0.82")
	t.Setenv("DIRECT_AUTO_FALLBACK", "true")
	t.Setenv("DIRECT_FALLBACK_ROUTE", "hybrid")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.DirectConfidenceThreshold != 0.82 {
		t.Fatalf("expected direct confidence threshold 0.82, got %v", cfg.DirectConfidenceThreshold)
	}
	if !cfg.DirectAutoFallback {
		t.Fatal("expected DIRECT_AUTO_FALLBACK=true")
	}
	if cfg.DirectFallbackRoute != "hybrid" {
		t.Fatalf("expected DIRECT_FALLBACK_ROUTE=hybrid, got %s", cfg.DirectFallbackRoute)
	}
}

func TestSplitOpenAIAPIKeys(t *testing.T) {
	clearConfigEnv(t)
	t.Setenv("USE_OPENAI", "true")
	t.Setenv("OPENAI_LLM_API_KEY", "llm-key")
	t.Setenv("OPENAI_EMBEDDING_API_KEY", "embedding-key")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load config with split keys: %v", err)
	}
	if cfg.OpenAILLMAPIKey != "llm-key" {
		t.Fatalf("expected llm key from OPENAI_LLM_API_KEY, got %q", cfg.OpenAILLMAPIKey)
	}
	if cfg.OpenAIEmbeddingAPIKey != "embedding-key" {
		t.Fatalf("expected embedding key from OPENAI_EMBEDDING_API_KEY, got %q", cfg.OpenAIEmbeddingAPIKey)
	}
}

func TestOpenAIAPIKeyFallbackForSplitKeys(t *testing.T) {
	clearConfigEnv(t)
	t.Setenv("USE_OPENAI", "true")
	t.Setenv("OPENAI_API_KEY", "shared-key")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load config with shared key fallback: %v", err)
	}
	if cfg.OpenAILLMAPIKey != "shared-key" {
		t.Fatalf("expected llm key fallback from OPENAI_API_KEY, got %q", cfg.OpenAILLMAPIKey)
	}
	if cfg.OpenAIEmbeddingAPIKey != "shared-key" {
		t.Fatalf("expected embedding key fallback from OPENAI_API_KEY, got %q", cfg.OpenAIEmbeddingAPIKey)
	}
}

func TestSQLDriverValidation(t *testing.T) {
	clearConfigEnv(t)
	t.Setenv("USE_OPENAI", "false")
	t.Setenv("SQL_DSN", "dummy")
	t.Setenv("SQL_DRIVER", SQLDriverPostgres)

	_, err := Load()
	if err != nil {
		t.Fatalf("expected valid postgres SQL_DRIVER, got err: %v", err)
	}
}

func TestInvalidSQLDriverRejected(t *testing.T) {
	clearConfigEnv(t)
	t.Setenv("USE_OPENAI", "false")
	t.Setenv("SQL_DSN", "dummy")
	t.Setenv("SQL_DRIVER", "oracle")

	_, err := Load()
	if err == nil {
		t.Fatal("expected invalid SQL_DRIVER to be rejected")
	}
}

func TestNoSQLDSNSkipsDriverValidation(t *testing.T) {
	clearConfigEnv(t)
	t.Setenv("USE_OPENAI", "false")
	t.Setenv("SQL_DSN", "")
	t.Setenv("SQL_DRIVER", "oracle")

	_, err := Load()
	if err != nil {
		t.Fatalf("expected no SQL validation when SQL_DSN empty, got err: %v", err)
	}
}

func TestLoadFromConfigFileThenEnvOverride(t *testing.T) {
	clearConfigEnv(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "defaults.json")
	content := `{
  "USE_OPENAI": false,
  "SERVER_ADDR": ":9000",
  "SQL_DRIVER": "sqlite",
  "SQL_DSN": ":memory:",
  "VECTOR_BACKEND": "memory"
}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp config file: %v", err)
	}

	t.Setenv("CONFIG_FILE", path)
	t.Setenv("SERVER_ADDR", ":9100")
	t.Setenv("SQL_DRIVER", SQLDriverMySQL)
	t.Setenv("SQL_DSN", "user:pass@tcp(localhost:3306)/db")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.ServerAddr != ":9100" {
		t.Fatalf("expected env override for SERVER_ADDR, got %s", cfg.ServerAddr)
	}
	if cfg.SQLDriver != SQLDriverMySQL {
		t.Fatalf("expected env override for SQL_DRIVER, got %s", cfg.SQLDriver)
	}
	if cfg.SQLDSN != "user:pass@tcp(localhost:3306)/db" {
		t.Fatalf("expected env override for SQL_DSN, got %s", cfg.SQLDSN)
	}
}

func TestLoadWithNestedEnvProfileAndEnvOverride(t *testing.T) {
	clearConfigEnv(t)
	dir := t.TempDir()
	configDir := filepath.Join(dir, "config")
	envDir := filepath.Join(configDir, "environments")
	if err := os.MkdirAll(envDir, 0o755); err != nil {
		t.Fatalf("mkdir env dir: %v", err)
	}

	base := `{
  "app": {"server_addr": ":8081", "log_level": "info"},
  "llm": {"use_openai": false, "model": "gpt-base"},
  "embedding": {"embedding_model": "embed-base", "embedding_dim": 128, "openai_embedding_base_url": "https://embed.example"},
  "rerank": {"url": "", "api_key": "", "model": "rerank-base", "top_m": 3},
  "web": {"serper_api_key": ""},
  "sql": {"driver": "sqlite", "dsn": ""},
  "vector": {
    "backend": "memory",
    "qdrant": {"host": "localhost", "grpc_port": 6334, "api_key": "", "use_tls": false},
    "collections": {"chunk": "chunks", "summary": "summaries"}
  },
  "retrieval": {"chunk_size": 700, "chunk_overlap": 100, "top_k": 5, "max_retries": 2},
  "routing": {"router_model": "router-a", "grade_model": "grader-a"},
  "timeouts": {"http_seconds": 20, "qdrant_seconds": 10}
}`
	if err := os.WriteFile(filepath.Join(configDir, "defaults.json"), []byte(base), 0o644); err != nil {
		t.Fatalf("write base file: %v", err)
	}

	dev := `{
  "app": {"server_addr": ":9090"},
  "rerank": {"url": "https://rerank.example/v1/rerank", "api_key": "dev-key"},
  "web": {"serper_api_key": "dev-serper"},
  "sql": {"driver": "pgx", "dsn": "postgres://dev:dev@localhost:5432/devdb?sslmode=disable"}
}`
	if err := os.WriteFile(filepath.Join(envDir, "dev.json"), []byte(dev), 0o644); err != nil {
		t.Fatalf("write env file: %v", err)
	}

	t.Setenv("CONFIG_FILE", filepath.Join(configDir, "defaults.json"))
	t.Setenv("APP_ENV", "dev")
	t.Setenv("RERANK_API_KEY", "env-key")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.ServerAddr != ":9090" {
		t.Fatalf("expected APP_ENV overlay server addr, got %s", cfg.ServerAddr)
	}
	if cfg.RerankURL != "https://rerank.example/v1/rerank" {
		t.Fatalf("expected APP_ENV overlay rerank url, got %s", cfg.RerankURL)
	}
	if cfg.RerankKey != "env-key" {
		t.Fatalf("expected env override for rerank key, got %s", cfg.RerankKey)
	}
	if cfg.SerperAPIKey != "dev-serper" {
		t.Fatalf("expected APP_ENV overlay serper key, got %s", cfg.SerperAPIKey)
	}
	if cfg.SQLDriver != SQLDriverPostgres {
		t.Fatalf("expected APP_ENV overlay sql driver, got %s", cfg.SQLDriver)
	}
}
