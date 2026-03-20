# Operations Notes

## Required env

- `OPENAI_LLM_API_KEY` (or fallback `OPENAI_API_KEY`)
- `OPENAI_EMBEDDING_API_KEY` (or fallback `OPENAI_API_KEY`)
- `OPENAI_LLM_BASE_URL` (or fallback `OPENAI_BASE_URL`)
- `OPENAI_EMBEDDING_BASE_URL` (or fallback `OPENAI_BASE_URL`)
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `QDRANT_HOST` + `QDRANT_GRPC_PORT` (when `VECTOR_BACKEND=qdrant`)

## Config loading order

服务启动按以下顺序加载配置：

1. 程序内置默认值
2. 基础配置文件（默认 `config/defaults.json`，可用 `CONFIG_FILE` 指定）
3. 分环境覆盖文件（`config/environments/<APP_ENV>.json`，默认 `APP_ENV=default`）
4. 环境变量覆盖（最终生效）

## Structured config layout

为避免所有配置平铺混杂，配置按域分组：

- `app`
- `llm`
- `embedding`
- `timeouts`
- `vector`
- `retrieval`
- `rerank`
- `routing`
- `sql`
- `web`
- `skills`

- `orchestration`

其中外部依赖（例如 `rerank.url`、`rerank.api_key`、`web.serper_api_key`）已纳入同一矩阵：
基础文件 + 分环境文件 + 环境变量覆盖。

## Orchestration knobs

- `PLANNER_MAX_SUBQUERIES`
- `ORCHESTRATOR_MAX_EXTERNAL_CALLS`
- `ORCHESTRATOR_TIMEOUT_SECONDS`
- `EARLY_STOP_MIN_CANDIDATES`
- `EARLY_STOP_TOP_SCORE`

## Skills

- `SKILLS_DIR`：技能目录，按 `<SKILLS_DIR>/<skillName>/SKILL.md` 组织。
- 当请求里显式传 `skill` 时，系统优先执行 skill；否则可根据问题与技能名匹配自动触发。
- `DIRECT_CONFIDENCE_THRESHOLD`
- `DIRECT_AUTO_FALLBACK`
- `DIRECT_FALLBACK_ROUTE` (`catalog|direct_chunk|hierarchical|hybrid|sql|web|direct`)

## SQL backend configuration

支持多数据库，通过 `SQL_DRIVER` + `SQL_DSN` 配置：

- `sqlite`（`modernc.org/sqlite`）
- `pgx`（`github.com/jackc/pgx/v5/stdlib`）
- `mysql`（`github.com/go-sql-driver/mysql`）
- `sqlserver`（`github.com/microsoft/go-mssqldb`）

示例：

- SQLite: `SQL_DRIVER=sqlite`, `SQL_DSN=:memory:`
- PostgreSQL: `SQL_DRIVER=pgx`, `SQL_DSN=postgres://user:pass@localhost:5432/dbname?sslmode=disable`
- MySQL: `SQL_DRIVER=mysql`, `SQL_DSN=user:pass@tcp(localhost:3306)/dbname?parseTime=true`
- SQL Server: `SQL_DRIVER=sqlserver`, `SQL_DSN=sqlserver://user:pass@localhost:1433?database=dbname`

## Observability

- JSON logs via `log/slog`
- `/health` endpoint for liveness
- `debug=true` in `/v1/ask` to inspect route/rewrite/grade traces

## Safety controls

- retry loop bounded by `MAX_RETRY_LOOPS`
- SQL tool only allows single `SELECT`
- web search requires `SERPER_API_KEY`

## Deployment

Use `docker compose up --build` for local deployment.

## Store adapter

- `internal/store/store.go` 定义统一 `VectorStore` 接口
- `internal/store/qdrant_store.go` 基于 `github.com/qdrant/go-client` 实现
- `internal/store/memory_store.go` 用于本地测试或降级运行
