"""
每个 turn 结束自动追加到 raw_memories.md。
带大小保护：超过上限时强制触发紧急截断。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from autocoder.memory.workspace import MemoryWorkspace

# raw_memories.md 紧急上限（防止整合失败导致无限增长）
RAW_MAX_CHARS = 60000


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

        # ★ 紧急保护：如果整合一直没触发导致 raw 过大，强制截断头部
        current = self.workspace.read_file("raw_memories.md")
        if len(current) > RAW_MAX_CHARS:
            truncated = current[-RAW_MAX_CHARS:]
            self.workspace.write_file(
                "raw_memories.md",
                "[...older turns truncated due to size limit...]\n" + truncated,
            )
            print("🗑️  [Memory] raw_memories.md exceeded limit, truncated head.")

    @property
    def turn_count(self) -> int:
        return self._turn_count