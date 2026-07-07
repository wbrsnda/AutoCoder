"""
错误自愈 - 拦截工具错误，生成 LLM 可理解的结构化反思提示。
让 Agent 能在下一轮自动修正，而不是死循环。
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class HealSuggestion:
    error_kind: str
    message: str
    suggested_actions: List[str]

    def to_llm_prompt(self) -> str:
        parts = [
            "⚠️ Tool Error Analysis:",
            f"- Error Type: {self.error_kind}",
            f"- Detail: {self.message}",
            "Suggested next actions:",
        ]
        for i, a in enumerate(self.suggested_actions, 1):
            parts.append(f"  {i}. {a}")
        return "\n".join(parts)


class SelfHealAnalyzer:
    """分析工具错误，产出结构化反思（对齐 Codex 的 reflect-and-retry 模式）"""

    ERROR_PATTERNS: Dict[str, Tuple[str, List[str]]] = {
        r"file not found|no such file": (
            "FILE_NOT_FOUND",
            [
                "Use mcp_list_dir to confirm the file exists",
                "Check the path is relative to workspace root",
                "If creating a new file, use mcp_write_file instead",
            ],
        ),
        r"original text not found": (
            "PATCH_CONTEXT_MISS",
            [
                "Use mcp_read_file to inspect current content first",
                "The 'original' text must match EXACTLY (whitespace, indentation)",
                "Consider mcp_write_file to fully rewrite the file if patch keeps failing",
            ],
        ),
        r"appears \d+ times": (
            "PATCH_AMBIGUOUS",
            [
                "Add more surrounding context to make the match unique",
                "Split into multiple smaller patches",
            ],
        ),
        r"permission denied|read-only": (
            "PERMISSION_DENIED",
            [
                "This operation is blocked by sandbox policy",
                "Report to user that the action requires elevated permissions; do NOT retry",
            ],
        ),
        r"blocked by hook|blocked:": (
            "HOOK_BLOCKED",
            [
                "This action violates a security rule",
                "Report the reason to Architect; do NOT retry the same call",
            ],
        ),
        r"missing required param|must be int|must be float": (
            "PARAM_INVALID",
            [
                "Recheck the tool signature and provide correct parameter names/types",
                "All required parameters must be present",
            ],
        ),
        r"timed out|timeout": (
            "TIMEOUT",
            [
                "The command took too long; split into smaller steps",
                "Check if the command waits for interactive input (not allowed)",
            ],
        ),
        r"duplicate call skipped": (
            "DUPLICATE_CALL",
            [
                "This exact call was already made in this turn",
                "Use the previous result instead; do NOT repeat the same call",
            ],
        ),
        r"tool not registered|tool not found": (
            "TOOL_NOT_FOUND",
            [
                "Use only tools listed in [VISIBLE TOOLS THIS TURN]",
                "Check the exact tool name spelling",
            ],
        ),
    }

    def analyze(self, tool_name: str, args: dict, error: str) -> Optional[HealSuggestion]:
        if not error:
            return None
        for pattern, (kind, actions) in self.ERROR_PATTERNS.items():
            if re.search(pattern, error, re.IGNORECASE):
                return HealSuggestion(
                    error_kind=kind,
                    message=f"Tool '{tool_name}' failed: {error[:200]}",
                    suggested_actions=actions,
                )
        return HealSuggestion(
            error_kind="UNKNOWN_ERROR",
            message=f"Tool '{tool_name}' returned: {error[:200]}",
            suggested_actions=[
                "Read the error message carefully",
                "Adjust arguments or try a different approach",
                "If it persists, report failure to Architect instead of retrying",
            ],
        )