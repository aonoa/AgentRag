package llm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	openaiembed "github.com/cloudwego/eino-ext/components/embedding/openai"
	openaimodel "github.com/cloudwego/eino-ext/components/model/openai"
	einoembedding "github.com/cloudwego/eino/components/embedding"
	einomodel "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

type Client interface {
	Chat(ctx context.Context, model string, systemPrompt string, userPrompt string) (string, error)
	ChatStream(ctx context.Context, model string, systemPrompt string, userPrompt string) (*schema.StreamReader[string], error)
	Embed(ctx context.Context, model string, texts []string) ([][]float64, error)
}

type OpenAIClient struct {
	chatModel      *openaimodel.ChatModel
	embedder       *openaiembed.Embedder
	llmModel       string
	embeddingModel string
}

func NewOpenAIClient(ctx context.Context, llmBaseURL, embeddingBaseURL, llmAPIKey, embeddingAPIKey, llmModel, embeddingModel string, timeout time.Duration, embeddingDim int) (*OpenAIClient, error) {
	chat, err := openaimodel.NewChatModel(ctx, &openaimodel.ChatModelConfig{
		APIKey:  llmAPIKey,
		BaseURL: llmBaseURL,
		Model:   llmModel,
		Timeout: timeout,
	})
	if err != nil {
		return nil, fmt.Errorf("create eino openai chat model: %w", err)
	}

	embedCfg := &openaiembed.EmbeddingConfig{
		APIKey:  embeddingAPIKey,
		BaseURL: embeddingBaseURL,
		Model:   embeddingModel,
		Timeout: timeout,
	}
	if embeddingDim > 0 {
		embedCfg.Dimensions = &embeddingDim
	}
	embedder, err := openaiembed.NewEmbedder(ctx, embedCfg)
	if err != nil {
		return nil, fmt.Errorf("create eino openai embedder: %w", err)
	}

	return &OpenAIClient{
		chatModel:      chat,
		embedder:       embedder,
		llmModel:       llmModel,
		embeddingModel: embeddingModel,
	}, nil
}

func (c *OpenAIClient) Chat(ctx context.Context, model string, systemPrompt string, userPrompt string) (string, error) {
	if model == "" {
		model = c.llmModel
	}
	if model == "" {
		return "", errors.New("chat model is empty")
	}

	out, err := c.chatModel.Generate(ctx, []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}, einomodel.WithModel(model))
	if err != nil {
		return "", fmt.Errorf("eino chat generate failed: %w", err)
	}
	if out == nil {
		return "", errors.New("empty chat response")
	}
	return strings.TrimSpace(out.Content), nil
}

func (c *OpenAIClient) ChatStream(ctx context.Context, model string, systemPrompt string, userPrompt string) (*schema.StreamReader[string], error) {
	if model == "" {
		model = c.llmModel
	}
	if model == "" {
		return nil, errors.New("chat model is empty")
	}

	stream, err := c.chatModel.Stream(ctx, []*schema.Message{
		schema.SystemMessage(systemPrompt),
		schema.UserMessage(userPrompt),
	}, einomodel.WithModel(model))
	if err != nil {
		return nil, fmt.Errorf("eino chat stream failed: %w", err)
	}
	return schema.StreamReaderWithConvert(stream, func(msg *schema.Message) (string, error) {
		if msg == nil {
			return "", schema.ErrNoValue
		}
		piece := strings.TrimSpace(msg.Content)
		if piece == "" {
			return "", schema.ErrNoValue
		}
		return msg.Content, nil
	}), nil
}

func (c *OpenAIClient) Embed(ctx context.Context, model string, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, errors.New("embed input is empty")
	}
	if model == "" {
		model = c.embeddingModel
	}
	if model == "" {
		return nil, errors.New("embedding model is empty")
	}

	result, err := c.embedder.EmbedStrings(ctx, texts, einoembedding.WithModel(model))
	if err != nil {
		return nil, fmt.Errorf("eino embedding failed: %w", err)
	}
	if len(result) != len(texts) {
		return nil, fmt.Errorf("embedding response size mismatch: got=%d want=%d", len(result), len(texts))
	}
	return result, nil
}

type MockClient struct{}

func NewMockClient() *MockClient {
	return &MockClient{}
}

func (m *MockClient) Chat(_ context.Context, _ string, systemPrompt string, userPrompt string) (string, error) {
	lower := strings.ToLower(userPrompt)
	if strings.Contains(lower, "route") || strings.Contains(systemPrompt, "ROUTE_SELECTOR") {
		switch {
		case strings.Contains(lower, "目录") || strings.Contains(lower, "有哪些") || strings.Contains(lower, "可用内容") || strings.Contains(lower, "browse") || strings.Contains(lower, "catalog"):
			return "catalog", nil
		case strings.Contains(lower, "sql") || strings.Contains(lower, "统计"):
			return "sql", nil
		case strings.Contains(lower, "实时") || strings.Contains(lower, "新闻") || strings.Contains(lower, "web"):
			return "web", nil
		case strings.Contains(lower, "概念") || strings.Contains(lower, "总结") || strings.Contains(lower, "overview"):
			return "hierarchical", nil
		default:
			return "direct_chunk", nil
		}
	}
	if strings.Contains(systemPrompt, "QUERY_REWRITE") {
		return strings.TrimSpace(userPrompt), nil
	}
	if strings.Contains(systemPrompt, "ANSWER_GRADER") {
		if strings.Contains(lower, "insufficient") {
			return `{"relevant":false,"score":0.2,"reason":"insufficient context"}`, nil
		}
		return `{"relevant":true,"score":0.9,"reason":"answer grounded in context"}`, nil
	}
	return "Mock answer:\n" + strings.TrimSpace(userPrompt), nil
}

func (m *MockClient) ChatStream(ctx context.Context, model string, systemPrompt string, userPrompt string) (*schema.StreamReader[string], error) {
	answer, err := m.Chat(ctx, model, systemPrompt, userPrompt)
	if err != nil {
		return nil, err
	}
	tokens := strings.Fields(answer)
	if len(tokens) == 0 {
		return schema.StreamReaderFromArray([]string{}), nil
	}
	out := make([]string, 0, len(tokens))
	for i, tk := range tokens {
		if i < len(tokens)-1 {
			out = append(out, tk+" ")
		} else {
			out = append(out, tk)
		}
	}
	return schema.StreamReaderFromArray(out), nil
}

func (m *MockClient) Embed(_ context.Context, _ string, texts []string) ([][]float64, error) {
	out := make([][]float64, 0, len(texts))
	for _, text := range texts {
		out = append(out, fakeEmbedding(text, 64))
	}
	return out, nil
}

func fakeEmbedding(input string, dim int) []float64 {
	vec := make([]float64, dim)
	if input == "" {
		return vec
	}
	for i, r := range []rune(input) {
		idx := i % dim
		vec[idx] += float64((int(r)%97)+1) / 100.0
	}
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vec
	}
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}
