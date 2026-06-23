from __future__ import annotations

from pathlib import Path

from autocoder.memory.models import MemoriesConfig

SUMMARY_CHAR_LIMIT = 2500 * 4


class MemoryInjector:
    def __init__(self, store, config: MemoriesConfig, workspace_dir: Path):
        self.store = store
        self.config = config
        self.workspace = workspace_dir

    def build_system_prompt_fragment(self) -> str:
        if not self.config.use_memories:
            return ""

        base_path = self.workspace / ".autocoder" / "memories"
        summary_path = base_path / "memory_summary.md"

        if not summary_path.exists():
            count = self.store.get_memory_count()
            if count == 0:
                return ""
            return (
                "\n## Memory\n\n"
                f"You have access to prior-run memories ({count} entries available). "
                "Use memory tools when prior context, conventions, preferences, or previous decisions may help.\n"
            )

        try:
            summary = summary_path.read_text(encoding="utf-8").strip()
        except Exception:
            summary = ""

        if not summary:
            return ""

        if len(summary) > SUMMARY_CHAR_LIMIT:
            summary = summary[:SUMMARY_CHAR_LIMIT] + "\n...[truncated]..."

        base = str(base_path).replace("\\", "/")
        return f"""
## Memory

You have access to a memory folder with guidance from prior runs. Use it whenever likely helpful.

Decision boundary:
- Skip memory only for clearly self-contained trivial requests.
- Use memory by default when the task may depend on prior workspace decisions, conventions, preferences, or older results.
- If the user asks what you remember, use memories_list / memories_search / memories_read.

Memory layout:
- {base}/memory_summary.md (already provided below; do NOT open again)
- {base}/MEMORY.md (searchable registry)
- {base}/rollout_summaries/
- {base}/skills/
- {base}/extensions/ad_hoc/notes/

Use memory tools when relevant:
- memories_list: list recent memories
- memories_search: search likely relevant prior context
- memories_read: read a specific memory in full
- add_ad_hoc_note: ONLY when the user explicitly asks you to update/save a memory

When answering from memory without current verification:
- briefly say it is memory-derived
- mention it may be stale if that matters

========= MEMORY_SUMMARY BEGINS =========
{summary}
========= MEMORY_SUMMARY ENDS =========
""".strip()

    def build_full_context(self, user_query: str) -> str:
        return self.build_system_prompt_fragment()