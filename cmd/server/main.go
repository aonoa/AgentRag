package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"agentragplus/internal/agent"
	"agentragplus/internal/api"
	"agentragplus/internal/config"
	"agentragplus/internal/ingest"
	"agentragplus/internal/llm"
	"agentragplus/internal/obs"
	"agentragplus/internal/rerank"
	"agentragplus/internal/retrieval"
	"agentragplus/internal/store"
	"agentragplus/internal/tools"

	_ "github.com/go-sql-driver/mysql"
	_ "github.com/jackc/pgx/v5/stdlib"
	_ "github.com/microsoft/go-mssqldb"
	_ "modernc.org/sqlite"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		panic(fmt.Errorf("load config: %w", err))
	}
	ctx := context.Background()

	logger := obs.New(cfg.LogLevel)
	shutdownTracing := obs.InitTracing(cfg.ObsEnableTracing)
	defer func() {
		_ = shutdownTracing(context.Background())
	}()
	obs.RegisterGlobalCallbackHandlerIfEnabled(cfg.ObsEnableCallbacks, logger)

	var llmClient llm.Client = llm.NewMockClient()
	if cfg.UseOpenAI {
		openAIClient, err := llm.NewOpenAIClient(ctx, cfg.OpenAILLMBaseURL, cfg.OpenAIEmbeddingBaseURL, cfg.OpenAILLMAPIKey, cfg.OpenAIEmbeddingAPIKey, cfg.LLMModel, cfg.EmbeddingModel, cfg.HTTPTimeout, cfg.EmbeddingDim)
		if err != nil {
			panic(fmt.Errorf("init eino openai clients: %w", err))
		}
		llmClient = openAIClient
		logger.Info("using OpenAI-compatible client", slog.String("llm_base_url", cfg.OpenAILLMBaseURL), slog.String("embedding_base_url", cfg.OpenAIEmbeddingBaseURL), slog.String("llm_model", cfg.LLMModel), slog.String("embed_model", cfg.EmbeddingModel))
	} else {
		logger.Warn("OpenAI keys missing or USE_OPENAI=false, using mock LLM/embedding client")
	}

	var vectorStore store.VectorStore
	switch cfg.VectorBackend {
	case config.VectorBackendQdrant:
		qdrantStore, err := store.NewQdrantStore(cfg.QdrantHost, cfg.QdrantGRPCPort, cfg.QdrantAPIKey, cfg.QdrantUseTLS, cfg.EmbeddingDim)
		if err != nil {
			panic(fmt.Errorf("init qdrant sdk store: %w", err))
		}
		vectorStore = qdrantStore
	default:
		vectorStore = store.NewMemoryStore()
		logger.Warn("using in-memory vector store (non-persistent)")
	}

	if err := vectorStore.EnsureCollections(ctx, cfg.ChunkCollection, cfg.SummaryCollection); err != nil {
		panic(fmt.Errorf("ensure vector collections: %w", err))
	}

	ingestSvc := ingest.NewService(cfg, llmClient, vectorStore)
	retriever := retrieval.NewRetriever(cfg, llmClient, vectorStore)
	reranker := rerank.NewHTTPClient(cfg)

	var db *sql.DB
	if cfg.SQLDSN != "" {
		db, err = sql.Open(cfg.SQLDriver, cfg.SQLDSN)
		if err != nil {
			panic(fmt.Errorf("open sql database: %w", err))
		}
		if err := db.PingContext(ctx); err != nil {
			panic(fmt.Errorf("ping sql database: %w", err))
		}
		defer db.Close()
	}
	sqlTool := tools.NewSQLTool(db, llmClient, cfg)
	webTool := tools.NewWebTool(cfg)

	agentSvc, err := agent.NewService(cfg, llmClient, retriever, reranker, sqlTool, webTool)
	if err != nil {
		panic(fmt.Errorf("build eino workflow: %w", err))
	}

	srv := api.NewServer(cfg, logger, ingestSvc, agentSvc)
	httpServer := &http.Server{
		Addr:              cfg.ServerAddr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		logger.Info("server started", slog.String("addr", cfg.ServerAddr))
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server crashed", slog.String("error", err.Error()))
			os.Exit(1)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Error("graceful shutdown failed", slog.String("error", err.Error()))
	}
}
