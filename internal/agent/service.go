package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"agentragplus/internal/config"
	"agentragplus/internal/domain"
	"agentragplus/internal/llm"
	"agentragplus/internal/rerank"
	"agentragplus/internal/retrieval"
	"agentragplus/internal/skills"
	"agentragplus/internal/tools"

	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

type Service struct {
	cfg       config.Config
	llm       llm.Client
	retriever *retrieval.Retriever
	reranker  rerank.Client
	sqlTool   *tools.SQLTool
	webTool   *tools.WebTool
	skills    *skills.Registry
	runnable  compose.Runnable[workflowInput, workflowOutput]
}

type AskStreamResult struct {
	Route             domain.RouteType
	RewrittenQuestion string
	Attempts          int
	References        []domain.Reference
	Debug             map[string]any
	Stream            *schema.StreamReader[string]
}

type workflowInput struct {
	Question     string
	Route        domain.RouteType
	Attempt      int
	SubQueries   []string
	ExtraFilters map[string]string
}

type workflowOutput struct {
	Answer            string
	RewrittenQuestion string
	References        []domain.Reference
	Retrieval         domain.RetrievalResult
	Context           string
	SystemPrompt      string
	UserPrompt        string
	Orchestration     map[string]any
}

type workflowState struct {
	Question          string
	Route             domain.RouteType
	Attempt           int
	SubQueries        []string
	ExtraFilters      map[string]string
	RewrittenQuestion string
	Retrieval         domain.RetrievalResult
	References        []domain.Reference
	Context           string
	SystemPrompt      string
	UserPrompt        string
	Answer            string
	Orchestration     map[string]any
}

type routeDecision struct {
	Route domain.RouteType
}

type gradePayload struct {
	Relevant bool    `json:"relevant"`
	Score    float64 `json:"score"`
	Reason   string  `json:"reason"`
}

type subQueryPlan struct {
	SubQueries []string
}

type retrievalOption struct {
	name  string
	route domain.RouteType
	cost  int
	local bool
	fetch func(context.Context) (domain.RetrievalResult, []domain.Reference, error)
}

type orchestrationStats struct {
	SubQueries           int
	ExecutedOptions      int
	ExternalCalls        int
	ExternalSkippedByCap int
	LocalEarlyStops      int
	FallbackTriggered    int
	Stages               []map[string]any
	StartAt              time.Time
}

func NewService(cfg config.Config, llmClient llm.Client, retr *retrieval.Retriever, rerankerClient rerank.Client, sqlTool *tools.SQLTool, webTool *tools.WebTool) (*Service, error) {
	svc := &Service{
		cfg:       cfg,
		llm:       llmClient,
		retriever: retr,
		reranker:  rerankerClient,
		sqlTool:   sqlTool,
		webTool:   webTool,
	}
	if strings.TrimSpace(cfg.SkillsDir) != "" {
		reg, regErr := skills.NewRegistryFromDir(cfg.SkillsDir)
		if regErr == nil {
			svc.skills = reg
		}
	}
	r, err := svc.buildWorkflow(context.Background())
	if err != nil {
		return nil, err
	}
	svc.runnable = r
	return svc, nil
}

func (s *Service) Ask(ctx context.Context, req domain.AskRequest) (domain.AskResponse, error) {
	question := strings.TrimSpace(req.Question)
	if question == "" {
		return domain.AskResponse{}, errors.New("question is required")
	}
	if s.shouldUseSkill(req) {
		return s.runSkillAsk(ctx, req)
	}
	route := req.ForceRoute
	if route == "" {
		rd, err := s.route(ctx, question)
		if err != nil {
			return domain.AskResponse{}, err
		}
		route = rd.Route
	}
	if route == domain.RouteDirect {
		system, user := s.finalAnswerPrompts(question, route, s.renderContext(nil))
		answer, err := s.llm.Chat(ctx, s.cfg.LLMModel, system, user)
		if err != nil {
			return domain.AskResponse{}, err
		}
		grade, err := s.grade(ctx, question, strings.TrimSpace(answer), s.renderContext(nil))
		if err != nil {
			return domain.AskResponse{}, err
		}
		if s.shouldFallbackDirect(grade) {
			fbRoute := domain.RouteType(strings.TrimSpace(strings.ToLower(s.cfg.DirectFallbackRoute)))
			if fbRoute != "" && fbRoute != domain.RouteDirect {
				stats := &orchestrationStats{SubQueries: 1, StartAt: time.Now()}
				r, refs, ferr := s.retrieveByRouteWithFallback(ctx, fbRoute, question, req.ExtraFilters, stats)
				if ferr == nil {
					ctxText := s.renderContext(r.Candidates)
					fs, fu := s.finalAnswerPrompts(question, fbRoute, ctxText)
					fAnswer, aerr := s.llm.Chat(ctx, s.cfg.LLMModel, fs, fu)
					if aerr == nil {
						resp := domain.AskResponse{
							Answer:            strings.TrimSpace(fAnswer),
							Route:             fbRoute,
							RewrittenQuestion: question,
							Attempts:          1,
							References:        refs,
						}
						if req.Debug {
							resp.Debug = map[string]any{
								"route": fbRoute,
								"direct_fallback": map[string]any{
									"triggered":       true,
									"from":            domain.RouteDirect,
									"to":              fbRoute,
									"grade":           grade,
									"confidence_gate": s.cfg.DirectConfidenceThreshold,
									"retrieval":       r.Debug,
								},
							}
						}
						return resp, nil
					}
				}
			}
		}
		resp := domain.AskResponse{
			Answer:            strings.TrimSpace(answer),
			Route:             route,
			RewrittenQuestion: question,
			Attempts:          1,
			References:        nil,
		}
		if req.Debug {
			resp.Debug = map[string]any{
				"route": route,
				"direct": map[string]any{
					"mode":              "direct_no_retrieval",
					"retrieval_skipped": true,
					"planner_skipped":   true,
					"rewrite_skipped":   true,
					"rerank_skipped":    true,
					"grader_skipped":    false,
					"grade":             grade,
					"confidence_gate":   s.cfg.DirectConfidenceThreshold,
					"fallback_enabled":  s.cfg.DirectAutoFallback,
				},
			}
		}
		return resp, nil
	}

	plannedSubQueries, err := s.planSubQueries(ctx, question)
	if err != nil {
		return domain.AskResponse{}, err
	}
	if len(plannedSubQueries) == 0 {
		plannedSubQueries = []string{question}
	}
	if s.cfg.PlannerMaxSubqueries > 0 && len(plannedSubQueries) > s.cfg.PlannerMaxSubqueries {
		plannedSubQueries = plannedSubQueries[:s.cfg.PlannerMaxSubqueries]
	}
	ctxOrch, cancel := s.withOrchestratorTimeout(ctx)
	defer cancel()

	current := question
	debug := map[string]any{"route": route}
	var refs []domain.Reference
	for attempt := 1; attempt <= s.cfg.MaxRetries; attempt++ {
		out, err := s.runnable.Invoke(ctxOrch, workflowInput{
			Question:     current,
			Route:        route,
			Attempt:      attempt,
			SubQueries:   plannedSubQueries,
			ExtraFilters: req.ExtraFilters,
		})
		if err != nil {
			return domain.AskResponse{}, err
		}
		refs = out.References
		grade, err := s.grade(ctx, out.RewrittenQuestion, out.Answer, out.Context)
		if err != nil {
			return domain.AskResponse{}, err
		}
		if !grade.Relevant && grade.Score < s.cfg.DirectConfidenceThreshold {
			fbRoute := domain.RouteType(strings.TrimSpace(strings.ToLower(s.cfg.DirectFallbackRoute)))
			if route == domain.RouteDirect && s.cfg.DirectAutoFallback && fbRoute != "" && fbRoute != route {
				route = fbRoute
				debug[fmt.Sprintf("attempt_%d", attempt)] = map[string]any{
					"rewritten":     out.RewrittenQuestion,
					"grade":         grade,
					"hits":          len(out.Retrieval.Candidates),
					"retrieval":     out.Retrieval.Debug,
					"orchestration": out.Orchestration,
					"fallback": map[string]any{
						"triggered": true,
						"from":      domain.RouteDirect,
						"to":        route,
					},
				}
				current = out.RewrittenQuestion
				continue
			}
		}

		debug[fmt.Sprintf("attempt_%d", attempt)] = map[string]any{
			"rewritten":     out.RewrittenQuestion,
			"grade":         grade,
			"hits":          len(out.Retrieval.Candidates),
			"retrieval":     out.Retrieval.Debug,
			"orchestration": out.Orchestration,
		}

		if grade.Relevant || attempt == s.cfg.MaxRetries {
			resp := domain.AskResponse{
				Answer:            out.Answer,
				Route:             route,
				RewrittenQuestion: out.RewrittenQuestion,
				Attempts:          attempt,
				References:        refs,
			}
			if req.Debug {
				resp.Debug = debug
			}
			return resp, nil
		}
		current = out.RewrittenQuestion + "\n请更具体并补充关键实体。"
	}

	return domain.AskResponse{}, errors.New("exhausted retry loop")
}

func (s *Service) AskStream(ctx context.Context, req domain.AskRequest) (AskStreamResult, error) {
	question := strings.TrimSpace(req.Question)
	if question == "" {
		return AskStreamResult{}, errors.New("question is required")
	}
	if s.shouldUseSkill(req) {
		return s.runSkillAskStream(ctx, req)
	}
	route := req.ForceRoute
	if route == "" {
		rd, err := s.route(ctx, question)
		if err != nil {
			return AskStreamResult{}, err
		}
		route = rd.Route
	}
	if route == domain.RouteDirect {
		system, user := s.finalAnswerPrompts(question, route, s.renderContext(nil))
		probe, err := s.llm.Chat(ctx, s.cfg.LLMModel, system, user)
		if err != nil {
			return AskStreamResult{}, err
		}
		grade, err := s.grade(ctx, question, strings.TrimSpace(probe), s.renderContext(nil))
		if err != nil {
			return AskStreamResult{}, err
		}
		if s.shouldFallbackDirect(grade) {
			fbRoute := domain.RouteType(strings.TrimSpace(strings.ToLower(s.cfg.DirectFallbackRoute)))
			if fbRoute != "" && fbRoute != domain.RouteDirect {
				stats := &orchestrationStats{SubQueries: 1, StartAt: time.Now()}
				r, refs, ferr := s.retrieveByRouteWithFallback(ctx, fbRoute, question, req.ExtraFilters, stats)
				if ferr == nil {
					ctxText := s.renderContext(r.Candidates)
					fs, fu := s.finalAnswerPrompts(question, fbRoute, ctxText)
					stream, serr := s.llm.ChatStream(ctx, s.cfg.LLMModel, fs, fu)
					if serr == nil {
						result := AskStreamResult{
							Route:             fbRoute,
							RewrittenQuestion: question,
							Attempts:          1,
							References:        refs,
							Stream:            stream,
						}
						if req.Debug {
							result.Debug = map[string]any{
								"route": fbRoute,
								"direct_fallback": map[string]any{
									"triggered":       true,
									"from":            domain.RouteDirect,
									"to":              fbRoute,
									"grade":           grade,
									"confidence_gate": s.cfg.DirectConfidenceThreshold,
									"retrieval":       r.Debug,
								},
							}
						}
						return result, nil
					}
				}
			}
		}
		stream, err := s.llm.ChatStream(ctx, s.cfg.LLMModel, system, user)
		if err != nil {
			return AskStreamResult{}, err
		}
		result := AskStreamResult{
			Route:             route,
			RewrittenQuestion: question,
			Attempts:          1,
			References:        nil,
			Stream:            stream,
		}
		if req.Debug {
			result.Debug = map[string]any{
				"route": route,
				"direct": map[string]any{
					"mode":              "direct_no_retrieval",
					"retrieval_skipped": true,
					"planner_skipped":   true,
					"rewrite_skipped":   true,
					"rerank_skipped":    true,
					"grader_skipped":    false,
					"grade":             grade,
					"confidence_gate":   s.cfg.DirectConfidenceThreshold,
					"fallback_enabled":  s.cfg.DirectAutoFallback,
				},
			}
		}
		return result, nil
	}
	plannedSubQueries, err := s.planSubQueries(ctx, question)
	if err != nil {
		return AskStreamResult{}, err
	}
	if len(plannedSubQueries) == 0 {
		plannedSubQueries = []string{question}
	}
	if s.cfg.PlannerMaxSubqueries > 0 && len(plannedSubQueries) > s.cfg.PlannerMaxSubqueries {
		plannedSubQueries = plannedSubQueries[:s.cfg.PlannerMaxSubqueries]
	}
	ctxOrch, cancel := s.withOrchestratorTimeout(ctx)
	defer cancel()

	current := question
	debug := map[string]any{"route": route}
	for attempt := 1; attempt <= s.cfg.MaxRetries; attempt++ {
		out, err := s.runnable.Invoke(ctxOrch, workflowInput{
			Question:     current,
			Route:        route,
			Attempt:      attempt,
			SubQueries:   plannedSubQueries,
			ExtraFilters: req.ExtraFilters,
		})
		if err != nil {
			return AskStreamResult{}, err
		}

		system, user := out.SystemPrompt, out.UserPrompt

		if attempt < s.cfg.MaxRetries {
			grade, err := s.grade(ctx, out.RewrittenQuestion, out.Answer, out.Context)
			if err != nil {
				return AskStreamResult{}, err
			}
			debug[fmt.Sprintf("attempt_%d", attempt)] = map[string]any{
				"rewritten":     out.RewrittenQuestion,
				"grade":         grade,
				"hits":          len(out.Retrieval.Candidates),
				"retrieval":     out.Retrieval.Debug,
				"orchestration": out.Orchestration,
			}
			if !grade.Relevant && grade.Score < s.cfg.DirectConfidenceThreshold {
				fbRoute := domain.RouteType(strings.TrimSpace(strings.ToLower(s.cfg.DirectFallbackRoute)))
				if route == domain.RouteDirect && s.cfg.DirectAutoFallback && fbRoute != "" && fbRoute != route {
					route = fbRoute
					current = out.RewrittenQuestion
					continue
				}
				current = out.RewrittenQuestion + "\n请更具体并补充关键实体。"
				continue
			}
		}

		stream, err := s.llm.ChatStream(ctx, s.cfg.LLMModel, system, user)
		if err != nil {
			return AskStreamResult{}, err
		}

		result := AskStreamResult{
			Route:             route,
			RewrittenQuestion: out.RewrittenQuestion,
			Attempts:          attempt,
			References:        out.References,
			Stream:            stream,
		}
		if req.Debug {
			result.Debug = debug
		}
		return result, nil
	}

	return AskStreamResult{}, errors.New("exhausted retry loop")
}

func (s *Service) prepareAttempt(ctx context.Context, route domain.RouteType, current string, attempt int, extraFilters map[string]string, plannedSubQueries []string, stats *orchestrationStats) (string, domain.RetrievalResult, []domain.Reference, string, error) {
	rewritten, err := s.rewrite(ctx, current, attempt)
	if err != nil {
		return "", domain.RetrievalResult{}, nil, "", err
	}
	retrievalResult, refs, err := s.retrieveForPlannedSubQueries(ctx, route, rewritten, plannedSubQueries, extraFilters, stats)
	if err != nil {
		return "", domain.RetrievalResult{}, nil, "", err
	}
	retrievalResult, err = s.rerankAndTrim(ctx, rewritten, retrievalResult)
	if err != nil {
		return "", domain.RetrievalResult{}, nil, "", err
	}
	contextText := s.renderContext(retrievalResult.Candidates)
	return rewritten, retrievalResult, refs, contextText, nil
}

func (s *Service) retrieveForPlannedSubQueries(ctx context.Context, route domain.RouteType, rewritten string, plannedSubQueries []string, extraFilters map[string]string, stats *orchestrationStats) (domain.RetrievalResult, []domain.Reference, error) {
	combined := make([]domain.RetrievalResult, 0, len(plannedSubQueries))
	allRefs := make([]domain.Reference, 0)
	for _, sq := range plannedSubQueries {
		query := strings.TrimSpace(sq)
		if query == "" {
			continue
		}
		res, refs, err := s.retrieveByRouteWithFallback(ctx, route, query, extraFilters, stats)
		if err != nil {
			return domain.RetrievalResult{}, nil, err
		}
		combined = append(combined, res)
		allRefs = append(allRefs, refs...)
	}
	if len(combined) == 0 {
		return domain.RetrievalResult{Candidates: nil, Debug: map[string]any{"mode": "empty"}}, nil, nil
	}
	if len(combined) == 1 {
		return combined[0], allRefs, nil
	}
	fused := fuseMultipleResultsRRF(combined, s.cfg.TopK)
	return fused, dedupeReferences(allRefs), nil
}

func (s *Service) retrieveByRouteWithFallback(ctx context.Context, route domain.RouteType, question string, extraFilters map[string]string, stats *orchestrationStats) (domain.RetrievalResult, []domain.Reference, error) {
	filter := make(map[string]any)
	for k, v := range extraFilters {
		filter[k] = v
	}

	options := s.buildOptionsForRoute(ctx, route, question, filter)
	if len(options) == 0 {
		return s.retrieveByRoute(ctx, route, question, extraFilters)
	}

	best := domain.RetrievalResult{Candidates: nil, Debug: map[string]any{"mode": "fallback"}}
	refs := make([]domain.Reference, 0)
	for _, opt := range options {
		if !opt.local && stats != nil && stats.ExternalCalls >= s.cfg.OrchestratorMaxExternal {
			stats.ExternalSkippedByCap++
			continue
		}
		start := time.Now()
		res, rfs, err := opt.fetch(ctx)
		if stats != nil {
			stats.ExecutedOptions++
			stage := map[string]any{
				"option":     opt.name,
				"local":      opt.local,
				"elapsed_ms": time.Since(start).Milliseconds(),
				"err":        err == nil,
				"hits":       len(res.Candidates),
			}
			stats.Stages = append(stats.Stages, stage)
		}
		if err != nil {
			continue
		}
		if !opt.local && stats != nil {
			stats.ExternalCalls++
		}
		if len(res.Candidates) > 0 {
			if best.Debug == nil {
				best.Debug = map[string]any{}
			}
			best = res
			best.Debug["selected_option"] = opt.name
			refs = rfs
			if opt.local && shouldEarlyStop(res, s.cfg.EarlyStopMinCandidates, s.cfg.EarlyStopTopScore) {
				if stats != nil {
					stats.LocalEarlyStops++
				}
				break
			}
			if !opt.local && stats != nil {
				stats.FallbackTriggered++
			}
		}
	}
	return best, refs, nil
}

func (s *Service) buildOptionsForRoute(ctx context.Context, route domain.RouteType, question string, filter map[string]any) []retrievalOption {
	baseLocalDense := retrievalOption{name: "local_dense", route: domain.RouteDirectChunk, cost: 1, local: true, fetch: func(ctx context.Context) (domain.RetrievalResult, []domain.Reference, error) {
		res, err := s.retriever.DirectChunk(ctx, question, filter)
		if err != nil {
			return domain.RetrievalResult{}, nil, err
		}
		return res, refsFromResult(res), nil
	}}
	baseLocalSparse := retrievalOption{name: "local_sparse", route: domain.RouteHybrid, cost: 1, local: true, fetch: func(ctx context.Context) (domain.RetrievalResult, []domain.Reference, error) {
		res, err := s.retriever.SparseChunk(ctx, question, filter)
		if err != nil {
			return domain.RetrievalResult{}, nil, err
		}
		return res, refsFromResult(res), nil
	}}
	localHier := retrievalOption{name: "local_hier", route: domain.RouteHierarchical, cost: 2, local: true, fetch: func(ctx context.Context) (domain.RetrievalResult, []domain.Reference, error) {
		res, err := s.retriever.Hierarchical(ctx, question, filter)
		if err != nil {
			return domain.RetrievalResult{}, nil, err
		}
		return res, refsFromResult(res), nil
	}}
	sqlOpt := retrievalOption{name: "sql", route: domain.RouteSQL, cost: 3, local: false, fetch: func(ctx context.Context) (domain.RetrievalResult, []domain.Reference, error) {
		text, err := s.sqlTool.Query(ctx, question)
		if err != nil {
			return domain.RetrievalResult{}, nil, err
		}
		res := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "sql_result", DocumentID: "sql", Source: "sql", Text: text, Score: 1, Layer: "sql"}}, Debug: map[string]any{"mode": "sql", "hits": 1}}
		return res, refsFromResult(res), nil
	}}
	webOpt := retrievalOption{name: "web", route: domain.RouteWeb, cost: 4, local: false, fetch: func(ctx context.Context) (domain.RetrievalResult, []domain.Reference, error) {
		text, err := s.webTool.Search(ctx, question)
		if err != nil {
			return domain.RetrievalResult{}, nil, err
		}
		res := domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "web_result", DocumentID: "web", Source: "web", Text: text, Score: 1, Layer: "web"}}, Debug: map[string]any{"mode": "web", "hits": 1}}
		return res, refsFromResult(res), nil
	}}

	switch route {
	case domain.RouteSQL:
		return []retrievalOption{baseLocalDense, baseLocalSparse, localHier, sqlOpt, webOpt}
	case domain.RouteWeb:
		return []retrievalOption{baseLocalDense, baseLocalSparse, localHier, webOpt}
	case domain.RouteHierarchical:
		return []retrievalOption{localHier, baseLocalDense, baseLocalSparse, sqlOpt, webOpt}
	case domain.RouteCatalog:
		return nil
	case domain.RouteHybrid:
		return []retrievalOption{{name: "hybrid_rrf", route: domain.RouteHybrid, cost: 1, local: true, fetch: func(ctx context.Context) (domain.RetrievalResult, []domain.Reference, error) {
			dense, err := s.retriever.DirectChunk(ctx, question, filter)
			if err != nil {
				return domain.RetrievalResult{}, nil, err
			}
			sparse, err := s.retriever.SparseChunk(ctx, question, filter)
			if err != nil {
				return domain.RetrievalResult{}, nil, err
			}
			merged := mergeCandidatesRRF(dense, sparse, s.cfg.TopK)
			return merged, refsFromResult(merged), nil
		}}, localHier, sqlOpt, webOpt}
	case domain.RouteDirect:
		return nil
	default:
		return []retrievalOption{baseLocalDense, baseLocalSparse, localHier, sqlOpt, webOpt}
	}
}

func (s *Service) planSubQueries(ctx context.Context, question string) ([]string, error) {
	system := "SUBQUERY_PLANNER: 将问题拆成最多3个子问题。只返回JSON数组字符串，例如 [\"子问题1\",\"子问题2\"]。如果无需拆分，返回包含原问题的数组。"
	out, err := s.llm.Chat(ctx, s.cfg.LLMModel, system, question)
	if err != nil {
		return nil, err
	}
	trim := strings.TrimSpace(out)
	if trim == "" {
		return []string{question}, nil
	}
	var subs []string
	if err := json.Unmarshal([]byte(trim), &subs); err != nil {
		return splitByPunct(question), nil
	}
	clean := make([]string, 0, len(subs))
	seen := make(map[string]bool)
	for _, s := range subs {
		t := strings.TrimSpace(s)
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		clean = append(clean, t)
		if len(clean) >= 3 {
			break
		}
	}
	if len(clean) == 0 {
		return []string{question}, nil
	}
	return clean, nil
}

func splitByPunct(question string) []string {
	parts := strings.FieldsFunc(question, func(r rune) bool {
		return r == '，' || r == ',' || r == '；' || r == ';' || r == '。' || r == '?' || r == '？'
	})
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{question}
	}
	return out
}

func refsFromResult(result domain.RetrievalResult) []domain.Reference {
	refs := make([]domain.Reference, 0, len(result.Candidates))
	for _, c := range result.Candidates {
		refs = append(refs, domain.Reference{DocumentID: c.DocumentID, ChunkID: c.ChunkID, Score: c.Score, Source: c.Source, Content: c.Text})
	}
	return refs
}

func dedupeReferences(refs []domain.Reference) []domain.Reference {
	idx := make(map[string]domain.Reference)
	for _, r := range refs {
		key := r.Source + "|" + r.ChunkID
		if old, ok := idx[key]; ok {
			if r.Score > old.Score {
				idx[key] = r
			}
			continue
		}
		idx[key] = r
	}
	out := make([]domain.Reference, 0, len(idx))
	for _, v := range idx {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > 20 {
		out = out[:20]
	}
	return out
}

func shouldEarlyStop(result domain.RetrievalResult, minCandidates int, topScoreThreshold float64) bool {
	if len(result.Candidates) == 0 {
		return false
	}
	if minCandidates > 0 && len(result.Candidates) >= minCandidates {
		return true
	}
	if topScoreThreshold > 0 && result.Candidates[0].Score >= topScoreThreshold {
		return true
	}
	return false
}

func fuseMultipleResultsRRF(results []domain.RetrievalResult, topK int) domain.RetrievalResult {
	k := 60.0
	rankMaps := make([]map[string]int, 0, len(results))
	idx := make(map[string]domain.RetrievalCandidate)
	for _, r := range results {
		rm := rankMap(r.Candidates)
		rankMaps = append(rankMaps, rm)
		for _, c := range r.Candidates {
			if _, ok := idx[c.ChunkID]; !ok {
				idx[c.ChunkID] = c
			}
		}
	}
	merged := make([]domain.RetrievalCandidate, 0, len(idx))
	for id, c := range idx {
		var score float64
		for _, rm := range rankMaps {
			if r, ok := rm[id]; ok {
				score += 1 / (k + float64(r))
			}
		}
		if math.IsNaN(score) || math.IsInf(score, 0) {
			score = 0
		}
		c.Score = score
		c.Layer = "rrf_multi"
		merged = append(merged, c)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	if topK > 0 && len(merged) > topK {
		merged = merged[:topK]
	}
	return domain.RetrievalResult{Candidates: merged, Debug: map[string]any{"mode": "multi_rrf", "sources": len(results), "merged_hits": len(merged)}}
}

func (s *Service) finalAnswerPrompts(question string, route domain.RouteType, contextText string) (string, string) {
	system := "你是企业知识库问答助手。优先使用上下文回答，回答要简洁准确，并在信息不足时明确说明。"
	user := fmt.Sprintf("问题：%s\n\n路由：%s\n\n上下文：\n%s\n\n请给出最终答案。", question, route, contextText)
	return system, user
}

func (s *Service) buildWorkflow(ctx context.Context) (compose.Runnable[workflowInput, workflowOutput], error) {
	wf := compose.NewWorkflow[workflowInput, workflowOutput]()

	initNode := compose.InvokableLambda(func(_ context.Context, in workflowInput) (workflowState, error) {
		subs := in.SubQueries
		if len(subs) == 0 {
			subs = []string{in.Question}
		}
		return workflowState{
			Question:     in.Question,
			Route:        in.Route,
			Attempt:      in.Attempt,
			SubQueries:   subs,
			ExtraFilters: in.ExtraFilters,
		}, nil
	})

	rewriteNode := compose.InvokableLambda(func(ctx context.Context, st workflowState) (workflowState, error) {
		rewritten, err := s.rewrite(ctx, st.Question, st.Attempt)
		if err != nil {
			return workflowState{}, err
		}
		st.RewrittenQuestion = rewritten
		return st, nil
	})

	retrieveNode := compose.InvokableLambda(func(ctx context.Context, st workflowState) (workflowState, error) {
		stats := &orchestrationStats{SubQueries: len(st.SubQueries), StartAt: time.Now()}
		res, refs, err := s.retrieveForPlannedSubQueries(ctx, st.Route, st.RewrittenQuestion, st.SubQueries, st.ExtraFilters, stats)
		if err != nil {
			return workflowState{}, err
		}
		st.Retrieval = res
		st.References = refs
		st.Orchestration = map[string]any{
			"subqueries":              stats.SubQueries,
			"executed_options":        stats.ExecutedOptions,
			"external_calls":          stats.ExternalCalls,
			"external_skipped_by_cap": stats.ExternalSkippedByCap,
			"local_early_stops":       stats.LocalEarlyStops,
			"fallback_triggered":      stats.FallbackTriggered,
			"elapsed_ms":              time.Since(stats.StartAt).Milliseconds(),
			"stages":                  stats.Stages,
		}
		return st, nil
	})

	rerankNode := compose.InvokableLambda(func(ctx context.Context, st workflowState) (workflowState, error) {
		out, err := s.rerankAndTrim(ctx, st.RewrittenQuestion, st.Retrieval)
		if err != nil {
			return workflowState{}, err
		}
		st.Retrieval = out
		st.References = refsFromResult(out)
		return st, nil
	})

	contextNode := compose.InvokableLambda(func(_ context.Context, st workflowState) (workflowState, error) {
		st.Context = s.renderContext(st.Retrieval.Candidates)
		return st, nil
	})

	promptNode := compose.InvokableLambda(func(_ context.Context, st workflowState) (workflowState, error) {
		system, user := s.finalAnswerPrompts(st.RewrittenQuestion, st.Route, st.Context)
		st.SystemPrompt = system
		st.UserPrompt = user
		return st, nil
	})

	answerNode := compose.InvokableLambda(func(ctx context.Context, st workflowState) (workflowState, error) {
		ans, err := s.llm.Chat(ctx, s.cfg.LLMModel, st.SystemPrompt, st.UserPrompt)
		if err != nil {
			return workflowState{}, err
		}
		msg := schema.AssistantMessage(ans, nil)
		st.Answer = strings.TrimSpace(msg.Content)
		return st, nil
	})

	outputNode := compose.InvokableLambda(func(_ context.Context, st workflowState) (workflowOutput, error) {
		return workflowOutput{
			Answer:            st.Answer,
			RewrittenQuestion: st.RewrittenQuestion,
			References:        st.References,
			Retrieval:         st.Retrieval,
			Context:           st.Context,
			SystemPrompt:      st.SystemPrompt,
			UserPrompt:        st.UserPrompt,
			Orchestration:     st.Orchestration,
		}, nil
	})

	wf.AddLambdaNode("init", initNode).AddInput(compose.START)
	wf.AddLambdaNode("rewrite", rewriteNode).AddInput("init")
	wf.AddLambdaNode("retrieve", retrieveNode).AddInput("rewrite")
	wf.AddLambdaNode("rerank", rerankNode).AddInput("retrieve")
	wf.AddLambdaNode("context", contextNode).AddInput("rerank")
	wf.AddLambdaNode("prompt", promptNode).AddInput("context")
	wf.AddLambdaNode("answer", answerNode).AddInput("prompt")
	wf.AddLambdaNode("output", outputNode).AddInput("answer")
	wf.AddEnd("output")

	return wf.Compile(ctx)
}

func (s *Service) route(ctx context.Context, question string) (routeDecision, error) {
	system := "ROUTE_SELECTOR: 你只输出一个枚举值：direct_chunk/catalog/hierarchical/sql/web/direct/hybrid。若问题是简单通用问答（无需知识库检索即可回答，例如寒暄、常识性定义、通用解释），输出 direct。若用户询问知识库目录、可用内容、有哪些文档、先浏览内容范围，输出 catalog。"
	msg, err := s.llm.Chat(ctx, s.cfg.RouterModel, system, question)
	if err != nil {
		return routeDecision{}, fmt.Errorf("route decision: %w", err)
	}
	val := domain.RouteType(strings.TrimSpace(strings.ToLower(msg)))
	switch val {
	case domain.RouteDirectChunk, domain.RouteCatalog, domain.RouteHierarchical, domain.RouteSQL, domain.RouteWeb, domain.RouteDirect, domain.RouteHybrid:
		return routeDecision{Route: val}, nil
	default:
		return routeDecision{Route: domain.RouteDirectChunk}, nil
	}
}

func (s *Service) rewrite(ctx context.Context, question string, attempt int) (string, error) {
	system := "QUERY_REWRITE: 重写问题以提升检索召回，保持语义一致。只返回改写后的问题。"
	user := fmt.Sprintf("第%d次尝试，原问题：%s", attempt, question)
	q, err := s.llm.Chat(ctx, s.cfg.LLMModel, system, user)
	if err != nil {
		return "", fmt.Errorf("query rewrite: %w", err)
	}
	q = strings.TrimSpace(q)
	if q == "" {
		return question, nil
	}
	return q, nil
}

func (s *Service) retrieveByRoute(ctx context.Context, route domain.RouteType, question string, extraFilters map[string]string) (domain.RetrievalResult, []domain.Reference, error) {
	filter := make(map[string]any)
	for k, v := range extraFilters {
		filter[k] = v
	}

	var result domain.RetrievalResult
	var err error
	switch route {
	case domain.RouteHierarchical:
		result, err = s.retriever.Hierarchical(ctx, question, filter)
	case domain.RouteSQL:
		text, e := s.sqlTool.Query(ctx, question)
		if e != nil {
			return domain.RetrievalResult{}, nil, e
		}
		result = domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "sql_result", DocumentID: "sql", Source: "sql", Text: text, Score: 1, Layer: "sql"}}, Debug: map[string]any{"mode": "sql", "hits": 1}}
	case domain.RouteWeb:
		text, e := s.webTool.Search(ctx, question)
		if e != nil {
			return domain.RetrievalResult{}, nil, e
		}
		result = domain.RetrievalResult{Candidates: []domain.RetrievalCandidate{{ChunkID: "web_result", DocumentID: "web", Source: "web", Text: text, Score: 1, Layer: "web"}}, Debug: map[string]any{"mode": "web", "hits": 1}}
	case domain.RouteCatalog:
		embed, e := s.llm.Embed(ctx, s.cfg.EmbeddingModel, []string{question})
		if e != nil {
			return domain.RetrievalResult{}, nil, fmt.Errorf("embed catalog question: %w", e)
		}
		summaries, e := s.retriever.SearchSummariesOnly(ctx, embed[0], s.cfg.TopK*2, filter)
		if e != nil {
			return domain.RetrievalResult{}, nil, e
		}
		result = domain.RetrievalResult{Candidates: summaries, Debug: map[string]any{"mode": "catalog", "summary_hits": len(summaries)}}
	case domain.RouteDirect:
		result = domain.RetrievalResult{Candidates: nil, Debug: map[string]any{"mode": "direct"}}
	case domain.RouteHybrid:
		dense, e := s.retriever.DirectChunk(ctx, question, filter)
		if e != nil {
			return domain.RetrievalResult{}, nil, e
		}
		sparse, e := s.retriever.SparseChunk(ctx, question, filter)
		if e != nil {
			return domain.RetrievalResult{}, nil, e
		}
		result = mergeCandidatesRRF(dense, sparse, s.cfg.TopK)
	default:
		result, err = s.retriever.DirectChunk(ctx, question, filter)
	}
	if err != nil {
		return domain.RetrievalResult{}, nil, err
	}
	refs := make([]domain.Reference, 0, len(result.Candidates))
	for _, c := range result.Candidates {
		refs = append(refs, domain.Reference{DocumentID: c.DocumentID, ChunkID: c.ChunkID, Score: c.Score, Source: c.Source, Content: c.Text})
	}
	return result, refs, nil
}

func (s *Service) rerankAndTrim(ctx context.Context, question string, in domain.RetrievalResult) (domain.RetrievalResult, error) {
	if len(in.Candidates) <= 1 {
		return in, nil
	}
	rows, err := s.reranker.Rerank(ctx, question, in.Candidates, s.cfg.RerankTopM)
	if err != nil {
		return domain.RetrievalResult{}, err
	}
	in.Candidates = rows
	if in.Debug == nil {
		in.Debug = map[string]any{}
	}
	in.Debug["reranked"] = true
	in.Debug["after_rerank"] = len(rows)
	return in, nil
}

func (s *Service) renderContext(cands []domain.RetrievalCandidate) string {
	if len(cands) == 0 {
		return "(no retrieval context)"
	}
	var b strings.Builder
	for i, c := range cands {
		b.WriteString(fmt.Sprintf("[%d] source=%s score=%.4f\n%s\n\n", i+1, c.Source, c.Score, c.Text))
	}
	return b.String()
}

func (s *Service) grade(ctx context.Context, question string, answer string, contextText string) (domain.GradeResult, error) {
	system := "ANSWER_GRADER: 你是评估器。仅输出JSON: {\"relevant\":bool,\"score\":0-1,\"reason\":\"...\"}"
	user := fmt.Sprintf("问题：%s\n\n答案：%s\n\n上下文：%s", question, answer, contextText)
	out, err := s.llm.Chat(ctx, s.cfg.GradeModel, system, user)
	if err != nil {
		return domain.GradeResult{}, fmt.Errorf("grade answer: %w", err)
	}
	var payload gradePayload
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &payload); err != nil {
		if strings.TrimSpace(answer) == "" {
			return domain.GradeResult{Relevant: false, Score: 0, Reason: "empty answer"}, nil
		}
		return domain.GradeResult{Relevant: true, Score: 0.5, Reason: "fallback parse"}, nil
	}
	return domain.GradeResult{Relevant: payload.Relevant, Score: payload.Score, Reason: payload.Reason}, nil
}

func mergeCandidatesRRF(dense domain.RetrievalResult, sparse domain.RetrievalResult, topK int) domain.RetrievalResult {
	k := 60.0
	denseRanks := rankMap(dense.Candidates)
	sparseRanks := rankMap(sparse.Candidates)

	index := map[string]domain.RetrievalCandidate{}
	for _, item := range dense.Candidates {
		index[item.ChunkID] = item
	}
	for _, item := range sparse.Candidates {
		if _, ok := index[item.ChunkID]; !ok {
			index[item.ChunkID] = item
		}
	}

	merged := make([]domain.RetrievalCandidate, 0, len(index))
	for id, item := range index {
		var fused float64
		if r, ok := denseRanks[id]; ok {
			fused += 1.0 / (k + float64(r))
		}
		if r, ok := sparseRanks[id]; ok {
			fused += 1.0 / (k + float64(r))
		}
		item.Score = fused
		item.Layer = "rrf_hybrid"
		merged = append(merged, item)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].Score > merged[j].Score })
	if topK > 0 && len(merged) > topK {
		merged = merged[:topK]
	}
	return domain.RetrievalResult{
		Candidates: merged,
		Debug: map[string]any{
			"mode":        "hybrid_rrf",
			"dense_hits":  len(dense.Candidates),
			"sparse_hits": len(sparse.Candidates),
			"merged_hits": len(merged),
			"rrf_k":       int(k),
		},
	}
}

func rankMap(items []domain.RetrievalCandidate) map[string]int {
	out := make(map[string]int, len(items))
	for i, item := range items {
		if _, ok := out[item.ChunkID]; !ok {
			out[item.ChunkID] = i + 1
		}
	}
	return out
}

func (s *Service) withOrchestratorTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	if s.cfg.OrchestratorTimeoutSeconds <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, time.Duration(s.cfg.OrchestratorTimeoutSeconds)*time.Second)
}

func (s *Service) shouldFallbackDirect(grade domain.GradeResult) bool {
	if !s.cfg.DirectAutoFallback {
		return false
	}
	if s.cfg.DirectConfidenceThreshold <= 0 {
		return false
	}
	if strings.TrimSpace(s.cfg.DirectFallbackRoute) == "" {
		return false
	}
	if grade.Relevant && grade.Score >= s.cfg.DirectConfidenceThreshold {
		return false
	}
	return grade.Score < s.cfg.DirectConfidenceThreshold
}

func (s *Service) shouldUseSkill(req domain.AskRequest) bool {
	if strings.TrimSpace(req.Skill) != "" {
		return true
	}
	if s.skills == nil {
		return false
	}
	_, ok := s.skills.SelectByQuestion(req.Question)
	return ok
}

func (s *Service) chooseSkill(req domain.AskRequest) (skills.Skill, bool) {
	if s.skills == nil {
		return skills.Skill{}, false
	}
	if name := strings.TrimSpace(req.Skill); name != "" {
		if !skills.IsValidSkillName(name) {
			return skills.Skill{}, false
		}
		return s.skills.Get(name)
	}
	return s.skills.SelectByQuestion(req.Question)
}

func (s *Service) runSkillAsk(ctx context.Context, req domain.AskRequest) (domain.AskResponse, error) {
	if strings.TrimSpace(req.Skill) != "" && !skills.IsValidSkillName(req.Skill) {
		return domain.AskResponse{}, errors.New("invalid skill name")
	}
	sk, ok := s.chooseSkill(req)
	if !ok {
		if strings.TrimSpace(req.Skill) != "" {
			return domain.AskResponse{}, errors.New("requested skill not found")
		}
		return domain.AskResponse{}, errors.New("no matching skill found")
	}
	question := strings.TrimSpace(req.Question)
	argsJSON := "{}"
	if len(req.SkillArgs) > 0 {
		b, err := json.Marshal(req.SkillArgs)
		if err == nil {
			argsJSON = string(b)
		}
	}
	system := "SKILL_ASSISTANT: 你必须基于给定的技能文档回答。若文档信息不足，明确说明不足。"
	user := fmt.Sprintf("技能名称：%s\n技能说明：%s\n技能目录：%s\n技能文档：\n%s\n\n用户问题：%s\n\n技能参数(JSON)：%s", sk.Name, sk.Description, sk.BaseDir, sk.Content, question, argsJSON)
	answer, err := s.llm.Chat(ctx, s.cfg.LLMModel, system, user)
	if err != nil {
		return domain.AskResponse{}, err
	}
	resp := domain.AskResponse{
		Answer:            strings.TrimSpace(answer),
		Route:             domain.RouteSkill,
		RewrittenQuestion: question,
		Attempts:          1,
		References: []domain.Reference{{
			DocumentID: "skill:" + sk.Name,
			ChunkID:    "skill_doc",
			Score:      1,
			Source:     sk.BaseDir,
			Content:    truncateText(sk.Content, 400),
		}},
	}
	if req.Debug {
		resp.Debug = map[string]any{
			"route": domain.RouteSkill,
			"skill": map[string]any{
				"name":        sk.Name,
				"description": sk.Description,
				"base_dir":    sk.BaseDir,
				"args":        req.SkillArgs,
			},
		}
	}
	return resp, nil
}

func (s *Service) runSkillAskStream(ctx context.Context, req domain.AskRequest) (AskStreamResult, error) {
	if strings.TrimSpace(req.Skill) != "" && !skills.IsValidSkillName(req.Skill) {
		return AskStreamResult{}, errors.New("invalid skill name")
	}
	sk, ok := s.chooseSkill(req)
	if !ok {
		if strings.TrimSpace(req.Skill) != "" {
			return AskStreamResult{}, errors.New("requested skill not found")
		}
		return AskStreamResult{}, errors.New("no matching skill found")
	}
	question := strings.TrimSpace(req.Question)
	argsJSON := "{}"
	if len(req.SkillArgs) > 0 {
		b, err := json.Marshal(req.SkillArgs)
		if err == nil {
			argsJSON = string(b)
		}
	}
	system := "SKILL_ASSISTANT: 你必须基于给定的技能文档回答。若文档信息不足，明确说明不足。"
	user := fmt.Sprintf("技能名称：%s\n技能说明：%s\n技能目录：%s\n技能文档：\n%s\n\n用户问题：%s\n\n技能参数(JSON)：%s", sk.Name, sk.Description, sk.BaseDir, sk.Content, question, argsJSON)
	stream, err := s.llm.ChatStream(ctx, s.cfg.LLMModel, system, user)
	if err != nil {
		return AskStreamResult{}, err
	}
	out := AskStreamResult{
		Route:             domain.RouteSkill,
		RewrittenQuestion: question,
		Attempts:          1,
		References: []domain.Reference{{
			DocumentID: "skill:" + sk.Name,
			ChunkID:    "skill_doc",
			Score:      1,
			Source:     sk.BaseDir,
			Content:    truncateText(sk.Content, 400),
		}},
		Stream: stream,
	}
	if req.Debug {
		out.Debug = map[string]any{
			"route": domain.RouteSkill,
			"skill": map[string]any{
				"name":        sk.Name,
				"description": sk.Description,
				"base_dir":    sk.BaseDir,
				"args":        req.SkillArgs,
			},
		}
	}
	return out, nil
}

func truncateText(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max]
}
