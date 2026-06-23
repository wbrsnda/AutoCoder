from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from autocoder.memory.models import MemoryItem, MemorySource, MemoriesConfig
from autocoder.memory.store import MemoryStore
from autocoder.memory.workspace import MemoryWorkspace


CONSOLIDATE_SYSTEM = """You are a Memory Writing Agent (Phase 2).

You receive raw memories and rollout summaries from prior sessions.
Your job is to consolidate them into:
1. MEMORY.md  -> searchable handbook / registry
2. memory_summary.md -> very compact summary injected into future system prompts

Return EXACTLY one JSON object with keys:
- memory_md
- memory_summary

Rules:
- memory_summary.md MUST start with the first line exactly: v1
- Prefer durable user preferences, repo facts, failure shields, workflow conventions,
  recurring constraints, and decision triggers.
- Do not invent facts.
- Do not store secrets.
"""


class PhaseTwoConsolidator:
    def __init__(
        self,
        llm: Optional[BaseChatModel],
        config: MemoriesConfig,
        store: MemoryStore,
    ):
        self.llm = llm
        self.config = config
        self.store = store
        self.workspace = MemoryWorkspace(store.workspace)

    async def run_consolidation(self, thread_id: str) -> bool:
        if not self.config.enable_phase2:
            return False

        if not self.workspace.ensure_initialized():
            return False

        candidates = self.store.get_stage1_for_consolidation(
            limit=self.config.max_raw_memories_for_consolidation,
            max_age_days=self.config.max_unused_days,
        )
        if not candidates:
            print("✅ [Memory] Phase2: no stage1 candidates")
            return True

        self._sync_workspace_inputs(candidates)

        diff_text = self.workspace.get_diff()
        if not self.workspace.has_changes():
            watermark = int(datetime.now().timestamp())
            self.store.mark_stage1_consolidated([m.id for m in candidates], watermark)
            return True

        self.workspace.write_workspace_diff(diff_text)

        memory_md, memory_summary = await self._build_outputs(candidates, diff_text)

        if not memory_summary.startswith("v1"):
            memory_summary = "v1\n" + memory_summary.lstrip()

        (self.workspace.root / "MEMORY.md").write_text(memory_md, encoding="utf-8")
        (self.workspace.root / "memory_summary.md").write_text(memory_summary, encoding="utf-8")

        watermark = int(datetime.now().timestamp())
        consolidated = MemoryItem(
            content=memory_summary,
            raw_memory=memory_md,
            rollout_summary=memory_summary,
            slug=f"consolidated-{watermark}",
            source=MemorySource.CONSOLIDATED,
            thread_id=thread_id,
            is_consolidated=True,
            consolidated_from=[m.id for m in candidates],
            consolidation_watermark=watermark,
            expires_at=datetime.now() + timedelta(days=self.config.max_memory_age_days * 2),
        )
        self.store.save_memory(consolidated)
        self.store.mark_stage1_consolidated([m.id for m in candidates], watermark)

        self.workspace.commit_baseline(f"memory phase2 consolidation {watermark}")
        print(f"✅ [Memory] Phase2 consolidated {len(candidates)} stage1 memories")
        return True

    def _sync_workspace_inputs(self, candidates: list[MemoryItem]) -> None:
        rollout_dir = self.workspace.rollout_summaries_dir
        rollout_dir.mkdir(parents=True, exist_ok=True)

        for mem in candidates:
            slug = mem.slug or mem.id
            path = rollout_dir / f"{slug}.md"
            body = (
                f"# {slug}\n\n"
                f"- rollout_id: {mem.id}\n"
                f"- rollout_path: {mem.rollout_path or 'unknown'}\n"
                f"- cwd: {mem.cwd}\n"
                f"- updated_at: {mem.source_updated_at.isoformat() if mem.source_updated_at else 'unknown'}\n\n"
                f"{mem.rollout_summary or mem.content}\n"
            )
            path.write_text(body, encoding="utf-8")

        raw_parts = ["# Raw Memories\n"]
        for mem in candidates:
            raw_parts.append(
                "\n---\n"
                f"id: {mem.id}\n"
                f"slug: {mem.slug}\n"
                f"cwd: {mem.cwd}\n"
                f"rollout_path: {mem.rollout_path or 'unknown'}\n"
                f"updated_at: {mem.source_updated_at.isoformat() if mem.source_updated_at else 'unknown'}\n"
                f"rollout_summary_file: rollout_summaries/{(mem.slug or mem.id)}.md\n"
                "---\n\n"
                f"{mem.raw_memory or mem.content}\n"
            )

        (self.workspace.root / "raw_memories.md").write_text("\n".join(raw_parts), encoding="utf-8")

    async def _build_outputs(self, candidates: list[MemoryItem], diff_text: str) -> tuple[str, str]:
        if self.llm is None:
            return self._build_outputs_fallback(candidates)

        payload = {
            "workspace_diff": diff_text[:200000],
            "raw_memories": [
                {
                    "id": m.id,
                    "slug": m.slug,
                    "cwd": m.cwd,
                    "rollout_path": m.rollout_path,
                    "rollout_summary": m.rollout_summary,
                    "raw_memory": m.raw_memory[:12000],
                }
                for m in candidates
            ],
            "existing_memory_md": self._safe_read(self.workspace.root / "MEMORY.md"),
            "existing_memory_summary": self._safe_read(self.workspace.root / "memory_summary.md"),
        }

        try:
            res = await self.llm.ainvoke([
                SystemMessage(content=CONSOLIDATE_SYSTEM),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])
            data = self._parse_json(str(res.content))
            memory_md = data.get("memory_md", "") or ""
            memory_summary = data.get("memory_summary", "") or ""
            if memory_md.strip() and memory_summary.strip():
                return memory_md, memory_summary
        except Exception as e:
            print(f"⚠️ [Memory] Phase2 LLM consolidation failed: {e}")

        return self._build_outputs_fallback(candidates)

    def _build_outputs_fallback(self, candidates: list[MemoryItem]) -> tuple[str, str]:
        lines = ["# MEMORY", "", "## Rollout Registry", ""]
        for mem in candidates:
            slug = mem.slug or mem.id
            one_line = self._first_nonempty_line(mem.rollout_summary or mem.content)
            lines.extend([
                f"### {slug}",
                f"- rollout_id: {mem.id}",
                f"- cwd: {mem.cwd}",
                f"- rollout_path: {mem.rollout_path or 'unknown'}",
                f"- rollout_summary_file: rollout_summaries/{slug}.md",
                f"- summary: {one_line}",
                "",
            ])
        memory_md = "\n".join(lines)

        summary_lines = ["v1", "", "Recent durable memory signals:"]
        for mem in candidates[:10]:
            one_line = self._first_nonempty_line(mem.rollout_summary or mem.content)
            summary_lines.append(f"- {one_line}")
        memory_summary = "\n".join(summary_lines)
        return memory_md, memory_summary

    def _parse_json(self, text: str) -> dict:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start:end + 1]
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("phase2 output is not a JSON object")
        return data

    def _safe_read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _first_nonempty_line(self, text: str) -> str:
        for line in text.splitlines():
            s = line.strip().lstrip("#").strip()
            if s:
                return s[:200]
        return "memory entry"