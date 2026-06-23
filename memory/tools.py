from __future__ import annotations
from datetime import datetime
from pathlib import Path
import re
from langchain_core.tools import tool
from autocoder.memory.workspace import MemoryWorkspace

_INTERNAL = {".git", "phase2_workspace_diff.md", ".baseline"}

def create_memory_tools(store, workspace_dir: Path):
    ws = MemoryWorkspace(workspace_dir)
    ws.ensure_initialized()
    root = ws.root

    def _resolve(p: str) -> Path:
        t = (root / p.lstrip("/\\")).resolve()
        if not str(t).startswith(str(root.resolve())):
            raise ValueError("outside memory root")
        return t

    @tool
    async def memories_list(path: str = "") -> str:
        """List memory files and directories."""
        target = _resolve(path) if path else root
        if not target.exists(): return f"Path not found: {path}"
        lines = []
        for p in sorted(target.iterdir()):
            if p.name in _INTERNAL: continue
            rel = p.relative_to(root).as_posix()
            lines.append(f"[DIR] {rel}/" if p.is_dir() else f"[FILE] {rel}")
        return "\n".join(lines) or "(empty)"

    @tool
    async def memories_search(queries: str, path: str = "") -> str:
        """Search memory files for keywords (grep)."""
        kws = [k.lower() for k in queries.split() if k.strip()]
        scope = _resolve(path) if path else root
        files = [scope] if scope.is_file() else list(root.rglob("*.md"))
        out = []
        for f in files:
            if f.name in _INTERNAL: continue
            try: lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except: continue
            for i,l in enumerate(lines):
                if any(k in l.lower() for k in kws):
                    out.append(f"{f.relative_to(root)}:{i+1}: {l.strip()}")
                    if len(out) > 50: return "\n".join(out)
        return "\n".join(out) if out else "No matches"

    @tool
    async def memories_read(path: str) -> str:
        """Read a memory file."""
        try:
            return _resolve(path).read_text(encoding="utf-8", errors="ignore")[:8000]
        except Exception as e:
            return f"Error: {e}"

    @tool
    async def add_ad_hoc_note(note: str, slug: str = "note") -> str:
        """Save explicit user memory. ONLY use when user says 'remember this'."""
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        safe = re.sub(r"[^a-z0-9-]", "-", slug.lower())[:30]
        p = ws.extensions_notes_dir / f"{ts}-{safe}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(note, encoding="utf-8")
        return f"Saved: {p.relative_to(root)}"

    return [memories_list, memories_search, memories_read, add_ad_hoc_note]