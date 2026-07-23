"""
上下文可见性网关 - 按需暴露工具给 LLM，避免 token 爆炸。
"""
from __future__ import annotations
from typing import Set
import re


TOOL_TAGS = {
    "mcp_list_dir": {"read", "discovery"},
    "mcp_read_file": {"read", "discovery"},
    "mcp_search_files": {"read", "discovery"},
    "mcp_find_files": {"read", "discovery"},
    "mcp_git_status": {"read", "git"},
    "mcp_git_diff": {"read", "git"},

    "mcp_write_file": {"write", "edit"},
    "mcp_append_file": {"write", "edit"},
    "mcp_apply_patch": {"write", "edit"},
    "mcp_delete_file": {"write", "destructive"},
    "mcp_write_files": {"write", "edit"},
    "mcp_create_directory": {"write"},
    "mcp_move_file": {"write"},
    "mcp_move_files": {"write"},

    "mcp_execute_bash": {"execute"},

    "memories_search": {"memory", "read"},
    "memories_read": {"memory", "read"},
    "memories_list": {"memory", "read"},
    "add_ad_hoc_note": {"memory", "write"},

    "rag_search": {"web", "read"},
}

READ_TOOLS = {"mcp_list_dir", "mcp_read_file", "mcp_search_files"}


class ContextGateway:
    """
    根据 delegation 语义选择性暴露工具。

    规则：
    1. delegation 中显式点名的工具必须暴露。
    2. 如果显式点名工具，不因为数量少而回退全量。
    3. 没有显式点名时，按意图标签选择。
    """

    def __init__(self, all_tools: list):
        self.all_tools = all_tools
        self.tool_map = {t.name: t for t in all_tools}

    def select_for_delegation(self, delegation: str) -> list:
        if not delegation or not delegation.strip():
            return self.all_tools

        d = delegation.lower()

        # 1. 显式工具名扫描
        explicit_names = {
            t.name for t in self.all_tools
            if t.name.lower() in d
        }

        if explicit_names:
            # 如果是写/删/执行工具，可额外暴露读工具作为辅助
            # 但 PlannerGuard 生成的 mcp_list_dir 只会暴露它自己。
            companion_names = set()
            if explicit_names & {
                "mcp_write_file", "mcp_append_file",
                "mcp_apply_patch", "mcp_delete_file",
                "mcp_execute_bash",
            }:
                companion_names |= READ_TOOLS

            names = explicit_names | companion_names
            return [
                self.tool_map[n]
                for n in sorted(names)
                if n in self.tool_map
            ]

        # 2. 意图标签匹配
        wanted_tags: Set[str] = set()

        if re.search(r"read|list|explore|find|search|check|show|inspect|列|读|看|查|找", d):
            wanted_tags.update({"read", "discovery"})
        if re.search(r"git|commit|diff|status|branch|提交|版本", d):
            wanted_tags.update({"git"})
        if re.search(r"write|create|edit|modify|patch|append|update|add|写|改|建|创建|追加|修改", d):
            wanted_tags.update({"write", "edit"})
        if re.search(r"delete|remove|clean|删", d):
            wanted_tags.update({"write", "destructive", "read"})
        if re.search(r"bash|shell|run|execute|command|npm|pip|git|test|执行|运行|测试", d):
            wanted_tags.update({"execute", "read"})
        if re.search(r"memor|记忆|记住|note", d):
            wanted_tags.update({"memory"})
        if re.search(r"web|documentation|docs|资料|文档|api reference|search|查询|搜索|查找|不确定|不知道|查一下", d):
            wanted_tags.update({"web"})

        if not wanted_tags:
            wanted_tags.update({"read", "discovery"})

        selected = []
        for tool in self.all_tools:
            tags = TOOL_TAGS.get(tool.name, {"read"})
            if tags & wanted_tags:
                selected.append(tool)

        return selected or self.all_tools

    def build_visible_manifest(self, tools: list, max_chars: int = 1500) -> str:
        lines = ["[VISIBLE TOOLS THIS TURN]"]
        used = 0

        for i, t in enumerate(tools):
            desc = ""
            if getattr(t, "description", None):
                desc = t.description.strip().splitlines()[0][:70]
            line = f"- {t.name}: {desc}"
            if used + len(line) > max_chars:
                lines.append(f"... (+{len(tools) - i} more tools available)")
                break
            lines.append(line)
            used += len(line)

        lines.append("Use ONLY these tools this turn.")
        return "\n".join(lines)