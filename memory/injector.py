"""
读取 memory_summary.md 注入 system prompt。
"""
from __future__ import annotations

from autocoder.memory.workspace import MemoryWorkspace


class MemoryInjector:
    def __init__(self, workspace: MemoryWorkspace):
        self.workspace = workspace

    def build_system_prompt_fragment(self) -> str:
        summary = self.workspace.read_file("memory_summary.md").strip()
        if not summary:
            return ""

        return (
            "\n\n[MEMORY_SUMMARY - 你从过往会话中学到的长期记忆]\n"
            f"{summary[:3000]}\n"
            "[END MEMORY_SUMMARY]"
        )