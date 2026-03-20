# AgentRagPlus 系统功能（完整概述）

本文档是对当前系统的完整概述，涵盖架构、特性、API、兼容性、配置、可观测性、安全控制和运维等方面。

---

## 1) 系统概述

AgentRagPlus 是一个基于 Eino 的 Agentic RAG 服务，其核心是一个工作流驱动的编排引擎。

主要目标：

- 多路由智能 QA（`direct`、`catalog`、`direct_chunk`、`hierarchical`、`hybrid`、`sql`、`web`、`skill`）

- 可靠的数据摄取管道（chunk/summarize/embed/store）

- 通过密集+稀疏混合和 RRF 融合提升检索质量

- 兼容 OpenAI 的聊天端点（包括流式传输和工具调用兼容性子集）

- 面向生产环境的控制措施（重试、回退、预算、超时、可观测性）

---

## 2) 高级架构

核心模块：

- `cmd/server` — 服务引导和依赖关系配置

- `internal/api` — HTTP API 层（`/health`、ingest、ask、stream、openai-compat）

- `internal/agent` — 编排、路由、工作流、评分/重试、回退

- `internal/ingest` — 文件规范化/分块/摘要/嵌入/存储写入

- `internal/retrieval` — 检索策略和摘要浏览

- `internal/rerank` — 带有优雅回退机制的外部重排序适配器

- `internal/llm` — Eino-Ext OpenAI 模型/嵌入客户端 + 模拟客户端

- `internal/store` — 向量存储适配器（`memory` + `qdrant` SDK 实现）

- `internal/tools` — SQL/Web 外部工具

- `internal/skills` — 技能注册表 (`<skill>/SKILL.md`)

- `internal/obs` — 跟踪/回调/日志关联辅助工具

---

## 3) 代理编排特性

### 3.1 工作流驱动的处理

代理处理基于图/工作流 (`compose.Workflow`)，包含以下节点阶段：

- `init`

- `rewrite`

- `retrieve`

- `rerank`

- `context`

- `prompt`

- `answer`

- `output`

这取代了手动临时链接，并提供了明确的执行边界。

### 3.2 路由类型

支持的路由枚举：

- `direct` — 直接回答路径，不进行检索

- `catalog` — 仅提供知识概览的摘要浏览路径

- `direct_chunk` — 直接密集检索

- `hierarchical` — 摘要->分块分层检索

- `hybrid` — 密集+稀疏融合，使用 RRF

- `sql` — 基于 SQL 工具的检索/查找

- `web` — 外部网络搜索

- `skill` — 基于技能的回答路径

### 3.3 直接置信度闭环（减少幻觉）

直接路径现在包含置信度校准：

1. 生成直接答案

2. 评估相关性/置信度

3. 如果低于阈值，则自动回退到轻量级检索路径（默认为 `catalog`）

4. 使用检索到的上下文重新回答

可通过以下方式配置：

- `DIRECT_CONFIDENCE_THRESHOLD`

- `DIRECT_AUTO_FALLBACK`

- `DIRECT_FALLBACK_ROUTE`

### 3.4 重试和评分循环

- 有界重试 (`MAX_RETRY_LOOPS`)

- 每次尝试的相关性评分

- 包含检索/编排遥测数据的尝试级调试有效载荷

### 3.5 多子查询编排

- 查询分解 (`planSubQueries`)

- 基于本地优先和受控回退的子查询路由执行

- 通过 RRF 进行多结果融合 (`multi_rrf`)

---

## 4) 检索功能

### 4.1 密集检索

- 基于嵌入的向量搜索，针对数据块集合

### 4.2 稀疏检索

- BM25 词汇评分（IDF/avgdl）

- 当模式支持时，使用 Qdrant 原生稀疏查询

- 兼容旧版仅密集检索集合的备用方案

### 4.3 混合检索

- `hybrid` 路由并行执行密集检索和稀疏检索

- 融合算法：**RRF** (k=60)

### 4.4 层级检索

- 优先进行摘要级召回

- 通过摘要 ID 扩展到块级候选结果

- 通过向量相似度进行轻量级重排序评分

### 4.5 目录浏览

- `catalog` 路由执行仅摘要检索

- 专为“有哪些内容可用？”而设计风格查询

- 适用于深度检索前的大型复杂知识库

---

## 5) 数据摄取管道

`/v1/ingest/upload` 管道：

1. 文件解析/规范化

2. 文档 ID 生成

3. 内容类别检测（叙述/表格）

4. 数据块分割（`CHUNK_SIZE`，`CHUNK_OVERLAP`）

5. 摘要生成

6. 数据块 + 摘要嵌入

7. 更新插入向量存储（摘要集合 + 数据块集合）

### Qdrant ID 兼容性强化

- 域 ID（例如 `sum_*`，`chk_*`）保留在有效负载中

- Qdrant 点 ID 是确定性的 UUID 兼容值

- 检索会将数据转换回有效负载业务 ID，以确保下游一致性

---

## 6) 技能能力

技能模式通过以下方式支持`SKILLS_DIR`：

- 目录约定：`<SKILLS_DIR>/<skillName>/SKILL.md`

- 可选的 YAML 前置元数据解析（`name`，`description`）

- 请求可以显式设置 `skill` 和 `skill_args`

- 如果显式提供了技能，则行为是确定性的（解决或出错）

- 可选的通过问题/名称匹配自动选择

安全性和一致性强化：

- 技能名称验证（`^[a-z0-9_-]+$`）

- 目录遍历/符号链接边界检查

- Ask/AskStream 行为与缺失/无效信息保持一致