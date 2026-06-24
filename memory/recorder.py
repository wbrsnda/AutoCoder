"""
每个 turn 结束自动追加到 raw_memories.md。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from autocoder.memory.workspace import MemoryWorkspace


class MemoryRecorder:
    def __init__(self, workspace: MemoryWorkspace):
        self.workspace = workspace
        self._turn_count = 0

    def record_turn(
        self,
        user_input: str,
        architect_response: str,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        """每个 turn 结束后追加 raw memory"""
        if not user_input or not architect_response:
            return

        self._turn_count += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        block = [
            f"\n---\n## Turn {self._turn_count} @ {ts}\n",
            f"**User**: {user_input.strip()[:500]}\n",
            f"**Architect Response**: {architect_response.strip()[:1000]}\n",
        ]

        if tool_calls:
            block.append("\n**Tools Used**:\n")
            for tc in tool_calls[:5]:
                name = tc.get("tool_name", "unknown")
                args = tc.get("tool_args", {})
                block.append(f"- `{name}` args={args}\n")

        self.workspace.append_file("raw_memories.md", "".join(block))

    @property
    def turn_count(self) -> int:
        return self._turn_count