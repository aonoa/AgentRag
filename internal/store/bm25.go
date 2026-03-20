package store

import (
	"hash/fnv"
	"math"
	"sort"
	"strings"
)

type bm25Doc struct {
	ID     string
	Tokens []string
}

type bm25Model struct {
	k1    float64
	b     float64
	idf   map[string]float64
	tf    map[string]map[string]int
	dl    map[string]float64
	avgdl float64
	N     float64
}

func buildBM25Model(docs []bm25Doc) bm25Model {
	m := bm25Model{
		k1:  1.2,
		b:   0.75,
		idf: make(map[string]float64),
		tf:  make(map[string]map[string]int),
		dl:  make(map[string]float64),
		N:   float64(len(docs)),
	}
	if len(docs) == 0 {
		return m
	}

	df := make(map[string]int)
	var totalLen float64
	for _, d := range docs {
		if len(d.Tokens) == 0 {
			m.tf[d.ID] = map[string]int{}
			m.dl[d.ID] = 0
			continue
		}
		tf := make(map[string]int)
		seen := make(map[string]bool)
		for _, tk := range d.Tokens {
			tf[tk]++
			if !seen[tk] {
				df[tk]++
				seen[tk] = true
			}
		}
		m.tf[d.ID] = tf
		docLen := float64(len(d.Tokens))
		m.dl[d.ID] = docLen
		totalLen += docLen
	}
	m.avgdl = totalLen / float64(len(docs))
	if m.avgdl <= 0 {
		m.avgdl = 1
	}

	for term, dfv := range df {
		n := float64(dfv)
		m.idf[term] = math.Log(1 + (m.N-n+0.5)/(n+0.5))
	}

	return m
}

func (m bm25Model) score(docID string, queryTokens []string) float64 {
	tf, ok := m.tf[docID]
	if !ok {
		return 0
	}
	dl := m.dl[docID]
	if dl <= 0 {
		dl = 1
	}
	avgdl := m.avgdl
	if avgdl <= 0 {
		avgdl = 1
	}

	var score float64
	for _, qt := range queryTokens {
		freq := float64(tf[qt])
		if freq <= 0 {
			continue
		}
		idf := m.idf[qt]
		den := freq + m.k1*(1-m.b+m.b*(dl/avgdl))
		score += idf * ((freq * (m.k1 + 1)) / den)
	}
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0
	}
	return score
}

func sparseTokens(query string) []string {
	lower := strings.ToLower(query)
	fields := strings.FieldsFunc(lower, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r >= 0x4e00 && r <= 0x9fff)
	})
	seen := make(map[string]bool, len(fields))
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, f)
	}
	return out
}

func sparseTermVector(text string) ([]uint32, []float32) {
	tokens := strings.Fields(strings.ToLower(text))
	if len(tokens) == 0 {
		return nil, nil
	}
	weights := make(map[uint32]float32)
	for _, tk := range tokens {
		tk = strings.TrimSpace(tk)
		if tk == "" {
			continue
		}
		id := tokenHashID(tk)
		weights[id] += 1
	}
	indices := make([]uint32, 0, len(weights))
	for id := range weights {
		indices = append(indices, id)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	values := make([]float32, 0, len(indices))
	for _, id := range indices {
		values = append(values, weights[id])
	}
	return indices, values
}

func tokenHashID(token string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(token))
	v := h.Sum32()
	if v == 0 {
		return 1
	}
	return v
}
