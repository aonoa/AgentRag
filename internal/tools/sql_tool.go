package tools

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"agentragplus/internal/config"
	"agentragplus/internal/llm"
)

type SQLTool struct {
	db  *sql.DB
	llm llm.Client
	cfg config.Config
}

func NewSQLTool(db *sql.DB, llmClient llm.Client, cfg config.Config) *SQLTool {
	return &SQLTool{db: db, llm: llmClient, cfg: cfg}
}

var selectPattern = regexp.MustCompile(`(?i)^\s*select\s+`)

func (t *SQLTool) Query(ctx context.Context, question string) (string, error) {
	if t.db == nil {
		return "", errors.New("sql tool not configured")
	}
	systemPrompt := "SQL_GENERATOR: 只允许生成单条 SELECT 语句，禁止修改数据。返回纯 SQL。"
	sqlText, err := t.llm.Chat(ctx, t.cfg.LLMModel, systemPrompt, question)
	if err != nil {
		return "", fmt.Errorf("generate sql: %w", err)
	}
	sqlText = strings.TrimSpace(strings.Trim(sqlText, "`"))
	if !selectPattern.MatchString(sqlText) {
		return "", fmt.Errorf("unsafe sql generated: %s", sqlText)
	}
	if strings.Contains(strings.ToLower(sqlText), ";") {
		return "", errors.New("multiple statements are not allowed")
	}

	rows, err := t.db.QueryContext(ctx, sqlText)
	if err != nil {
		return "", fmt.Errorf("execute sql: %w", err)
	}
	defer rows.Close()

	columns, err := rows.Columns()
	if err != nil {
		return "", fmt.Errorf("read columns: %w", err)
	}

	result := make([]string, 0)
	result = append(result, strings.Join(columns, " | "))
	for rows.Next() {
		values := make([]any, len(columns))
		ptrs := make([]any, len(columns))
		for i := range values {
			ptrs[i] = &values[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return "", fmt.Errorf("scan row: %w", err)
		}
		parts := make([]string, 0, len(values))
		for _, v := range values {
			parts = append(parts, fmt.Sprintf("%v", v))
		}
		result = append(result, strings.Join(parts, " | "))
		if len(result) >= 21 {
			break
		}
	}
	if len(result) == 1 {
		result = append(result, "(no rows)")
	}
	return strings.Join(result, "\n"), nil
}
