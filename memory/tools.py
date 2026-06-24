"""
对齐 Codex: 4 个文件操作工具。
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from autocoder.memory.workspace import MemoryWorkspace


def _slug(text: str) -> str:
    text = text.lower().strip().replace(" ", "-")
    text = re.sub(r"[^a-z0-9\-_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:50] or "note"


def create_memory_tools(workspace_dir: Path):
    workspace = MemoryWorkspace(workspace_dir)
    workspace.ensure_initialized()

    @tool
    async def memories_search(query: str) -> str:
        """
        Search memory files for keywords using grep.
        Use this to find prior context, user preferences, or past decisions.
        """
        if not query.strip():
            return "Error: query is empty"

        results = workspace.grep(query, max_results=20)
        if not results:
            return f"No memories matching: {query}"

        by_file = {}
        for r in results:
            by_file.setdefault(r["file"], []).append(f"  L{r['line']}: {r['content']}")

        lines = [f"Found {len(results)} match(es):", ""]
        for f, matches in by_file.items():
            lines.append(f"--- {f} ---")
            lines.extend(matches[:8])
            lines.append("")
        return "\n".join(lines)

    @tool
    async def memories_read(path: str) -> str:
        """
        Read a memory file by relative path.
        Common paths: 'MEMORY.md', 'memory_summary.md', 'raw_memories.md'
        """
        if not path.strip():
            return "Error: path is empty"

        content = workspace.read_file(path)
        if not content:
            return f"File not found or empty: {path}"
        return content[:8000]

    @tool
    async def memories_list(path: str = "") -> str:
        """
        List memory files and directories.
        Use path="" for root, or path="rollout_summaries" for subdirectory.
        """
        files = workspace.list_files(sub_path=path)
        if not files:
            return f"(empty: {path or 'root'})"
        return "\n".join(files)

    @tool
    async def add_ad_hoc_note(content: str, slug: str = "note") -> str:
        """
        Save a memory note. ONLY use when user explicitly says "记住这个" / "remember this".
        """
        if not content.strip():
            return "Error: content is empty"

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        file_stem = f"{ts}-{_slug(slug)}"
        workspace.write_file(
            f"extensions/ad_hoc/notes/{file_stem}.md",
            content.strip() + "\n"
        )
        workspace.commit_all(f"ad_hoc_note: {file_stem}")
        return f"✅ Saved memory note: {file_stem}.md"

    return [memories_search, memories_read, memories_list, add_ad_hoc_note]