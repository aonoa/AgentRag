package skills

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsValidSkillName(t *testing.T) {
	if !IsValidSkillName("eino-guide") {
		t.Fatal("expected valid skill name")
	}
	if IsValidSkillName("../escape") {
		t.Fatal("expected invalid traversal-like skill name")
	}
	if IsValidSkillName("bad name") {
		t.Fatal("expected invalid skill name with space")
	}
}

func TestRegistrySkipsInvalidDirectories(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "good_skill"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "good_skill", "SKILL.md"), []byte("---\nname: good_skill\n---\nhello"), 0o644); err != nil {
		t.Fatalf("write skill: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "bad skill"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bad skill", "SKILL.md"), []byte("---\nname: bad skill\n---\nignored"), 0o644); err != nil {
		t.Fatalf("write skill: %v", err)
	}

	r, err := NewRegistryFromDir(dir)
	if err != nil {
		t.Fatalf("new registry: %v", err)
	}
	if _, ok := r.Get("good_skill"); !ok {
		t.Fatal("expected good_skill loaded")
	}
	if _, ok := r.Get("bad skill"); ok {
		t.Fatal("expected invalid skill directory to be ignored")
	}
}
