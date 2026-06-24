"""
对齐 Codex: 纯文件 + Git 管理。
只依赖标准库 + subprocess (git)。
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class MemoryWorkspace:
    def __init__(self, workspace_dir: Path):
        self.root = workspace_dir / ".autocoder" / "memories"
        self.rollout_summaries_dir = self.root / "rollout_summaries"
        self.extensions_notes_dir = self.root / "extensions" / "ad_hoc" / "notes"
        self.skills_dir = self.root / "skills"

    def ensure_initialized(self) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.rollout_summaries_dir.mkdir(parents=True, exist_ok=True)
            self.extensions_notes_dir.mkdir(parents=True, exist_ok=True)
            self.skills_dir.mkdir(parents=True, exist_ok=True)

            if not (self.root / ".git").exists():
                self._run_git(["init", "-q"])
                self._run_git(["config", "user.email", "autocoder@local"])
                self._run_git(["config", "user.name", "AutoCoder Memory"])
                (self.root / ".gitkeep").write_text("", encoding="utf-8")
                self._run_git(["add", "."])
                self._run_git(["commit", "-m", "init memory workspace", "--allow-empty"])
                print(f"✅ [Memory] Git workspace initialized at {self.root}")
            return True
        except Exception as e:
            print(f"⚠️  [Memory] Workspace init failed: {e}")
            return False

    # ── 写 ──────────────────────────────────────────
    def write_file(self, rel_path: str, content: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def append_file(self, rel_path: str, content: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)

    # ── 读 ──────────────────────────────────────────
    def read_file(self, rel_path: str) -> str:
        path = self.root / rel_path
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def exists(self, rel_path: str) -> bool:
        return (self.root / rel_path).exists()

    # ── 搜 ──────────────────────────────────────────
    def grep(self, query: str, max_results: int = 30) -> list[dict]:
        """简单 grep 搜索所有 .md 文件"""
        terms = [t.lower() for t in query.split() if t.strip()]
        if not terms:
            return []

        results = []
        for file in self.root.rglob("*.md"):
            if ".git" in file.parts:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            rel = str(file.relative_to(self.root)).replace("\\", "/")
            for i, line in enumerate(text.splitlines(), 1):
                line_lower = line.lower()
                if any(t in line_lower for t in terms):
                    results.append({
                        "file": rel,
                        "line": i,
                        "content": line.strip()[:200],
                    })
                    if len(results) >= max_results:
                        return results
        return results

    def list_files(self, sub_path: str = "") -> list[str]:
        target = self.root / sub_path if sub_path else self.root
        if not target.exists() or not target.is_dir():
            return []
        items = []
        for p in sorted(target.iterdir()):
            if p.name == ".git" or p.name == ".gitkeep":
                continue
            rel = str(p.relative_to(self.root)).replace("\\", "/")
            items.append(f"[DIR] {rel}/" if p.is_dir() else f"[FILE] {rel}")
        return items

    # ── Git ─────────────────────────────────────────
    def commit_all(self, message: str) -> None:
        try:
            self._run_git(["add", "."])
            self._run_git(["commit", "-m", message, "--allow-empty"])
        except Exception:
            pass

    def has_changes(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> None:
        subprocess.run(
            ["git"] + args,
            cwd=str(self.root),
            check=True,
            capture_output=True,
        )