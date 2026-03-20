package skills

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

type FrontMatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

type Skill struct {
	Name        string
	Description string
	Content     string
	BaseDir     string
}

type Registry struct {
	baseDir string
	skills  map[string]Skill
}

var validSkillNameRe = regexp.MustCompile(`^[a-z0-9_-]+$`)

func IsValidSkillName(name string) bool {
	trim := strings.ToLower(strings.TrimSpace(name))
	if trim == "" {
		return false
	}
	return validSkillNameRe.MatchString(trim)
}

func NewRegistryFromDir(baseDir string) (*Registry, error) {
	trim := strings.TrimSpace(baseDir)
	if trim == "" {
		return nil, errors.New("skills base dir is empty")
	}
	entries, err := os.ReadDir(trim)
	if err != nil {
		return nil, fmt.Errorf("read skills base dir: %w", err)
	}
	r := &Registry{baseDir: trim, skills: map[string]Skill{}}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if !IsValidSkillName(e.Name()) {
			continue
		}
		skillDir := filepath.Join(trim, e.Name())
		resolvedSkillDir, err := filepath.EvalSymlinks(skillDir)
		if err != nil {
			continue
		}
		rel, err := filepath.Rel(trim, resolvedSkillDir)
		if err != nil || strings.HasPrefix(rel, "..") || strings.HasPrefix(filepath.ToSlash(rel), "../") {
			continue
		}
		skillPath := filepath.Join(skillDir, "SKILL.md")
		raw, err := os.ReadFile(skillPath)
		if err != nil {
			continue
		}
		skill, err := parseSkillFile(e.Name(), skillDir, string(raw))
		if err != nil {
			continue
		}
		r.skills[strings.ToLower(skill.Name)] = skill
	}
	return r, nil
}

func (r *Registry) Get(name string) (Skill, bool) {
	if r == nil {
		return Skill{}, false
	}
	trim := strings.ToLower(strings.TrimSpace(name))
	if !IsValidSkillName(trim) {
		return Skill{}, false
	}
	s, ok := r.skills[trim]
	return s, ok
}

func (r *Registry) List() []Skill {
	if r == nil {
		return nil
	}
	out := make([]Skill, 0, len(r.skills))
	for _, s := range r.skills {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (r *Registry) SelectByQuestion(question string) (Skill, bool) {
	if r == nil {
		return Skill{}, false
	}
	q := strings.ToLower(strings.TrimSpace(question))
	if q == "" {
		return Skill{}, false
	}
	for _, s := range r.skills {
		name := strings.ToLower(s.Name)
		if strings.Contains(q, name) {
			return s, true
		}
	}
	return Skill{}, false
}

func parseSkillFile(dirName, skillDir, raw string) (Skill, error) {
	content := strings.TrimSpace(raw)
	fm := FrontMatter{}
	if strings.HasPrefix(content, "---") {
		parts := strings.SplitN(content, "---", 3)
		if len(parts) == 3 {
			if err := yaml.Unmarshal([]byte(parts[1]), &fm); err != nil {
				return Skill{}, fmt.Errorf("parse skill front matter: %w", err)
			}
			content = strings.TrimSpace(parts[2])
		}
	}
	name := strings.TrimSpace(fm.Name)
	if name == "" {
		name = dirName
	}
	desc := strings.TrimSpace(fm.Description)
	if desc == "" {
		desc = "Skill loaded from SKILL.md"
	}
	return Skill{Name: name, Description: desc, Content: content, BaseDir: skillDir}, nil
}
