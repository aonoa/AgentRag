# AgentRagPlus (Eino-based Agentic RAG)

基于 **CloudWeGo Eino** 与 `RAG开发文档.md` 落地的 Agentic RAG 服务。

## 核心能力

- 智能路由：`direct_chunk / hierarchical / sql / web / direct / hybrid`
- 查询改写：进入检索前由 LLM 改写
- 检索：稠密检索（向量）+ 稀疏检索（BM25 lexical，含IDF/avgdl）
- 重排序：调用外部 rerank 接口（未配置时回退分数截断）
- 生成：Eino workflow 调度生成答案
- 评估与重试：相关性评分 + 最大重试循环
- 入库：上传文件后切块、摘要、embedding 并写入向量库

### Hybrid 检索说明（已实现）

- `RouteHybrid` 现在执行：**Dense + Sparse** 双路召回
  - Dense: 向量检索（embedding + vector store）
  - Sparse: lexical 检索（BM25）
- 融合算法：**RRF (Reciprocal Rank Fusion)**
  - `score(doc) = Σ 1 / (k + rank_i)`
  - 当前 `k = 60`

复杂问题编排说明：

- 对多知识点问题，系统会先做子问题拆分（最多3个）
- 每个子问题执行分阶段检索：本地优先（dense/sparse/hierarchical）→ 外部回退（sql/web）
- 多子问题结果使用 RRF 二次融合（`multi_rrf`）

预算与早停控制：

- `PLANNER_MAX_SUBQUERIES`：每次请求最多子问题数
- `ORCHESTRATOR_MAX_EXTERNAL_CALLS`：每次请求允许外部检索调用上限（SQL/Web）
- `ORCHESTRATOR_TIMEOUT_SECONDS`：检索编排总超时
- `EARLY_STOP_MIN_CANDIDATES`：本地检索命中达到该候选数即早停
- `EARLY_STOP_TOP_SCORE`：本地检索最高分超过阈值即早停

开启 `debug=true` 时，响应中会包含 orchestration telemetry：

- `subqueries` / `executed_options` / `external_calls`
- `external_skipped_by_cap` / `local_early_stops` / `fallback_triggered`
- `elapsed_ms` / `stages`（每阶段耗时与命中）

Qdrant 路径说明：

- 默认使用 Qdrant 命名向量：`dense` + `sparse`
- 稀疏检索优先使用 Qdrant 原生 `sparse` query
- 若集合为旧 schema（仅 dense），自动回退到 scroll + 本地 BM25 计算（保证兼容）

## 为什么现在使用 Eino-Ext 而不是手写 HTTP

LLM 和 Embedding 已迁移到 **Eino-Ext OpenAI 组件**：

- `github.com/cloudwego/eino-ext/components/model/openai`
- `github.com/cloudwego/eino-ext/components/embedding/openai`

这样可以保证：

1. 与 Eino 生态一致（消息结构、模型选项、callbacks）
2. 避免重复实现底层协议细节
3. 后续切换模型供应商时改动更小

## 快速开始

1. 复制配置

```bash
cp .env.example .env
```

配置优先级：

1. 内置默认值（代码）
2. 基础文件 `config/defaults.json`（或 `CONFIG_FILE` 指定路径）
3. 分环境覆盖 `config/environments/<APP_ENV>.json`（默认 `APP_ENV=default`）
4. 环境变量（最高优先级，覆盖前面所有层）

OpenAI 端点可拆分配置：

- `OPENAI_LLM_BASE_URL`：仅用于 LLM/chat
- `OPENAI_EMBEDDING_BASE_URL`：仅用于 Embedding
- `OPENAI_BASE_URL`：兼容兜底（会同时设置两者）

OpenAI Key 也支持拆分：

- `OPENAI_LLM_API_KEY`：仅用于 LLM/chat
- `OPENAI_EMBEDDING_API_KEY`：仅用于 Embedding
- `OPENAI_API_KEY`：兼容兜底（会同时设置两者）

2. 启动依赖与服务

```bash
docker compose up --build
```

3. 健康检查

```bash
curl http://localhost:8080/health
```

4. 上传文档

```bash
curl -X POST http://localhost:8080/v1/ingest/upload \
  -F "file=@RAG开发文档.md"
```

5. 提问

```bash
curl -X POST http://localhost:8080/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"系统支持哪些检索策略？","debug":true}'
```

6. OpenAI 兼容问答（Chat Completions）

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "系统支持哪些检索策略？"}
    ]
  }'
```

兼容说明：

- 已兼容基础 Chat Completions 请求/响应格式（支持流式 SSE）
- 支持 `stream=true`
- token usage 为近似估算值（用于兼容字段）

OpenAI 流式示例：

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4-mini",
    "stream": true,
    "messages": [
      {"role": "user", "content": "请用三句话介绍这个RAG系统"}
    ]
  }'
```

原生 RAG 流式示例：

```bash
curl -N -X POST http://localhost:8080/v1/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"系统支持哪些检索策略？"}'
```

## 目录结构

- `cmd/server`：启动入口
- `internal/api`：HTTP 接口
- `internal/agent`：Eino workflow + 路由/改写/评估循环
- `internal/ingest`：入库流水线
- `internal/retrieval`：检索策略
- `internal/rerank`：重排适配
- `internal/tools`：SQL/Web 工具
- `internal/llm`：Eino-Ext LLM/Embedding 封装 + Mock
- `internal/store`：向量存储适配层（Memory + Qdrant SDK，可扩展其他后端）

## 向量库适配说明

- Qdrant 使用官方 SDK：`github.com/qdrant/go-client`
- `VectorStore` 作为统一适配接口，业务层（ingest/retrieval）不感知具体数据库
- 未来替换 Milvus/PGVector/ES 时，仅需新增 store 实现并在启动层切换

## 开发验证

```bash
go test ./...
go build ./...
```

## Makefile 工作流

```bash
make run-default   # 使用 APP_ENV=default
make run-local     # 使用 APP_ENV=local
make run-dev       # 使用 APP_ENV=dev 覆盖
make run-staging   # 使用 APP_ENV=staging 覆盖
make run-prod      # 使用 APP_ENV=prod 覆盖
make test          # 运行测试
make build         # 编译
```

## 配置结构化分层

配置文件已按业务域分层，相关配置放在同一层级：

- `app`：服务监听与日志
- `llm`：模型与对话侧 OpenAI 配置
- `embedding`：Embedding 模型与 Embedding OpenAI 配置
- `timeouts`：网络超时
- `vector`：向量存储与Qdrant
- `retrieval`：切块与检索参数
- `rerank`：重排外部依赖（`url/api_key/model/top_m`）
- `routing`：路由与评估模型
- `sql`：SQL数据源
- `web`：Web search外部依赖（`serper_api_key`）
- `orchestration`：编排控制（含 direct 置信度阈值与自动回退）
- `skills`：技能目录（`base_dir`）

## SQL 数据库可配置

系统 SQL Tool 已支持多数据库（通过 `database/sql` 驱动切换）：

- `SQL_DRIVER=sqlite`
- `SQL_DRIVER=pgx`
- `SQL_DRIVER=mysql`
- `SQL_DRIVER=sqlserver`

当 `SQL_DSN` 为空时，SQL Tool 自动禁用。
