"""
PlannerGuard / Architect Output Guard

职责：
1. 防止 Architect 对明显需要工具的请求直接 AWAITING USER INPUT。
2. 对简单明确的 delegation 生成确定性 tool_call。
3. 支持从 FileTracker 的目录缓存中解析“那个 tex 文件”这类指代。
4. 保护多行写入 delegation，不截断 mcp_write_file / patch / append 的 payload。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


@dataclass
class DelegationPlan:
    instruction: str
    tool_name: str
    args: Dict[str, Any]
    reason: str


class PlannerGuard:
    FILE_RE = re.compile(
        r"([A-Za-z0-9_\-./\\]+?\.(?:py|txt|md|json|yaml|yml|toml|ini|csv|js|ts|tsx|jsx|html|css|env|tex))",
        re.IGNORECASE,
    )

    DESTRUCTIVE_RE = re.compile(
        r"(删除|删掉|清空|delete|remove|rm\s+|wipe|clean)",
        re.IGNORECASE,
    )

    WORKSPACE_LIST_RE = re.compile(
        r"("
        r"当前工作目录|工作目录|workspace|目录下|项目里|有什么文件|有哪些文件|"
        r"list files|what files|show files|what'?s in|目录内容|文件列表"
        r")",
        re.IGNORECASE,
    )

    LIST_INTENT_RE = re.compile(
        r"(有什么|有哪些|列出|看看|看一下|告诉我|show|list|what|inspect|找不到)",
        re.IGNORECASE,
    )

    READ_INTENT_RE = re.compile(
        r"(读取|读一下|查看|打开|看看|read|show|inspect|cat)",
        re.IGNORECASE,
    )

    SEARCH_INTENT_RE = re.compile(
        r"(搜索|查找|grep|search|find)",
        re.IGNORECASE,
    )

    TEX_REF_RE = re.compile(
        r"(\.tex|tex文件|latex|LaTeX|那个tex|那个 tex)",
        re.IGNORECASE,
    )

    FIX_INTENT_RE = re.compile(
        r"(语法错误|错误|修复|修改|改一下|fix|syntax|compile|编译|找不到)",
        re.IGNORECASE,
    )

    WRITE_TOOL_RE = re.compile(
        r"(mcp_write_file|mcp_append_file|mcp_apply_patch)",
        re.IGNORECASE,
    )

    def __init__(self, tool_names: Optional[List[str]] = None, file_tracker=None):
        self.tool_names = set(tool_names or [])
        self.file_tracker = file_tracker

    # ─────────────────────────────────────────────
    # Architect 保护
    # ─────────────────────────────────────────────

    def plan_for_user_request(
        self,
        user_text: str,
        has_coder_report: bool = False,
    ) -> Optional[DelegationPlan]:
        """
        根据用户原始请求判断是否必须使用工具。
        只处理高置信度场景。
        """
        if not user_text or has_coder_report:
            return None

        text = user_text.strip()

        # 破坏性请求不自动 delegate，必须走 Architect 确认流程
        if self.DESTRUCTIVE_RE.search(text):
            return None

        # 1. 当前工作目录 / workspace 有什么文件
        if self.WORKSPACE_LIST_RE.search(text) and self.LIST_INTENT_RE.search(text):
            if self._tool_available("mcp_list_dir"):
                return DelegationPlan(
                    instruction="Use mcp_list_dir to list the workspace root.",
                    tool_name="mcp_list_dir",
                    args={"directory": "."},
                    reason="User asks to inspect workspace files.",
                )

        # 2. “那个 tex 文件 / tex 文件有语法错误”这类指代
        if self.TEX_REF_RE.search(text) and self.FIX_INTENT_RE.search(text):
            tex_file = self._find_unique_file_by_ext(".tex")
            if tex_file and self._tool_available("mcp_read_file"):
                return DelegationPlan(
                    instruction=f'Use mcp_read_file to read "{tex_file}".',
                    tool_name="mcp_read_file",
                    args={"file_path": tex_file},
                    reason="User references the unique .tex file in workspace.",
                )

            # 不知道是哪一个 tex，就先列目录
            if self._tool_available("mcp_list_dir"):
                return DelegationPlan(
                    instruction="Use mcp_list_dir to list the workspace root and identify the .tex file.",
                    tool_name="mcp_list_dir",
                    args={"directory": "."},
                    reason="User references a .tex file but exact path is unknown.",
                )

        # 3. 读取明确文件
        file_path = self._extract_file_path(text)
        if file_path and self.READ_INTENT_RE.search(text):
            if self._tool_available("mcp_read_file"):
                return DelegationPlan(
                    instruction=f'Use mcp_read_file to read "{file_path}".',
                    tool_name="mcp_read_file",
                    args={"file_path": file_path},
                    reason="User asks to read a specific file.",
                )

        # 4. 搜索类请求不强猜 regex
        if self.SEARCH_INTENT_RE.search(text):
            return None

        return None

    def pre_delegate(
        self,
        user_text: str,
        has_coder_report: bool = False,
    ) -> Optional[str]:
        plan = self.plan_for_user_request(user_text, has_coder_report)
        return plan.instruction if plan else None

    def normalize_architect_output(
        self,
        user_text: str,
        content: str,
        has_coder_report: bool = False,
    ) -> str:
        """
        修正 Architect 违规输出。

        重点：
        - 如果已有 DELEGATE，保留 delegation。
        - 对 mcp_write_file / append / patch 的多行 payload 绝不截断。
        - 如果同时出现 AWAITING，只移除 AWAITING 之后的内容。
        """
        content = (content or "").strip()
        if not content:
            return content

        if "DELEGATE TO CODER:" in content:
            delegation = self._extract_delegation(content)
            if delegation:
                return f"DELEGATE TO CODER: {delegation}"

        forced = self.pre_delegate(user_text, has_coder_report)
        if forced and not has_coder_report:
            return f"DELEGATE TO CODER: {forced}"

        return content

    # ─────────────────────────────────────────────
    # Coder 快路径
    # ─────────────────────────────────────────────

        # ─────────────────────────────────────────────
    # Coder 保护：简单 delegation 直接转 tool_call
    # ─────────────────────────────────────────────

    # 匹配 "create/overwrite/write/update/append X.ext" 中的文件名
    _WRITE_TARGET_RE = re.compile(
        r"(?:create|overwrite|write(?:\s+to)?|update|append(?:\s+to)?|to)\s+"
        r"['\"`]?([A-Za-z0-9_\-./\\]+\.[a-zA-Z0-9]+)['\"`]?",
        re.IGNORECASE,
    )
    _FILE_PATH_KW_RE = re.compile(
        r"file_path\s*[=:]\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    def _extract_write_target(self, text: str) -> Optional[str]:
        """从 delegation 里抽取写入目标文件路径。"""
        m = self._FILE_PATH_KW_RE.search(text)
        if m:
            return m.group(1).strip().replace("\\", "/")
        m = self._WRITE_TARGET_RE.search(text)
        if m:
            path = m.group(1).strip().strip("\"'`")
            if path.startswith("./"):
                path = path[2:]
            return path.replace("\\", "/")
        return self._extract_file_path(text)

    def _extract_code_block(self, text: str) -> Optional[str]:
        """
        从 delegation 中抽取 ``` 包裹的代码块。
        用 first-open + last-close 的贪婪策略，正确处理 Markdown 内嵌代码块。
        """
        open_m = re.search(r"```[a-zA-Z0-9_+.\-]*[ \t]*\r?\n", text)
        if not open_m:
            # 也允许没有语言标记的 ```
            open_m = re.search(r"```[ \t]*\r?\n", text)
            if not open_m:
                return None
        start = open_m.end()
        end = text.rfind("```")
        if end <= start:
            return None
        return text[start:end].rstrip("\n").rstrip("\r")

    def parse_delegation_to_tool_call(self, delegation: str) -> Optional[Dict[str, Any]]:
        """
        把简单明确的 delegation 转成 LangChain AIMessage.tool_calls 格式。
        这可以绕过弱模型不稳定的 tool-calling 行为。
        """
        if not delegation:
            return None

        text = delegation.strip()
        low = text.lower()

        # mcp_list_dir
        if "mcp_list_dir" in low:
            directory = self._extract_kwarg(text, ["directory", "dir", "path"]) or "."
            return self._tool_call("mcp_list_dir", {"directory": directory})

        # mcp_read_file
        if "mcp_read_file" in low:
            fp = (
                self._extract_kwarg(text, ["file_path", "path", "file"])
                or self._extract_file_path(text)
            )
            if fp:
                return self._tool_call("mcp_read_file", {"file_path": fp})

        # mcp_search_files
        if "mcp_search_files" in low:
            regex = self._extract_kwarg(text, ["regex", "query", "pattern"])
            file_pattern = self._extract_kwarg(text, ["file_pattern"]) or "*.*"
            if regex:
                return self._tool_call(
                    "mcp_search_files",
                    {"regex": regex, "file_pattern": file_pattern},
                )

        # memories_search / memories_read / memories_list
        if "memories_search" in low:
            query = self._extract_kwarg(text, ["query", "q"])
            if query:
                return self._tool_call("memories_search", {"query": query})
        if "memories_read" in low:
            path = self._extract_kwarg(text, ["path", "file"])
            if path:
                return self._tool_call("memories_read", {"path": path})
        if "memories_list" in low:
            path = self._extract_kwarg(text, ["path"]) or ""
            return self._tool_call("memories_list", {"path": path})

        # mcp_delete_file
        if "mcp_delete_file" in low:
            fp = (
                self._extract_kwarg(text, ["file_path", "path", "file"])
                or self._extract_file_path(text)
            )
            if fp:
                return self._tool_call("mcp_delete_file", {"file_path": fp})

        # ★ 新增：mcp_write_file / mcp_append_file
        # Architect 通常输出：Use mcp_write_file to create X.py with the following content:\n```lang\n<code>\n```
        if "mcp_write_file" in low or "mcp_append_file" in low:
            tool = "mcp_write_file" if "mcp_write_file" in low else "mcp_append_file"
            fp = self._extract_write_target(text)
            content = self._extract_code_block(text)
            # 允许 content 为空字符串（写空文件），但不允许 None
            if fp and content is not None:
                return self._tool_call(tool, {"file_path": fp, "content": content})
            # 有文件名但没代码块 → 退给 Coder LLM 处理
            return None

        return None

    # ─────────────────────────────────────────────
    # 内部工具
    # ─────────────────────────────────────────────

    def _tool_available(self, tool_name: str) -> bool:
        return not self.tool_names or tool_name in self.tool_names

    def _tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": name,
            "args": args,
            "id": f"harness_{uuid.uuid4().hex[:8]}",
        }

    def _extract_file_path(self, text: str) -> Optional[str]:
        m = self.FILE_RE.search(text)
        if not m:
            return None
        path = m.group(1).strip().strip("\"'")
        if path.startswith("./"):
            path = path[2:]
        return path.replace("\\", "/")

    def _extract_quoted_path(self, text: str) -> Optional[str]:
        m = re.search(r'["\']([^"\']+\.[A-Za-z0-9]+)["\']', text)
        if not m:
            return None
        path = m.group(1).strip()
        if path.startswith("./"):
            path = path[2:]
        return path.replace("\\", "/")

    def _extract_kwarg(self, text: str, names: List[str]) -> Optional[str]:
        for name in names:
            m = re.search(
                rf"{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                return m.group(1).strip()

            m = re.search(
                rf"{re.escape(name)}\s*:\s*['\"]([^'\"]+)['\"]",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                return m.group(1).strip()
        return None

    def _extract_delegation(self, content: str) -> str:
        if "DELEGATE TO CODER:" not in content:
            return ""

        raw = content.split("DELEGATE TO CODER:", 1)[-1]

        # 如果模型违反规则同时输出 AWAITING，则只删除 AWAITING 及其后续
        raw = raw.split("AWAITING USER INPUT")[0].strip()

        # 写入/patch 类 delegation 可能携带多行 payload，必须完整保留
        if self.WRITE_TOOL_RE.search(raw) or "```" in raw:
            return raw

        # 其他普通 delegation 只取第一条指令，避免模型多写解释
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[0] if lines else raw

    def _known_files_from_dir_listings(self) -> List[str]:
        """
        从 FileTracker 的目录缓存中提取已知文件名。
        兼容你当前 FileTracker._dir_listings: dict[str, tuple[str, float]]
        """
        if not self.file_tracker:
            return []

        listings = getattr(self.file_tracker, "_dir_listings", {}) or {}
        files: List[str] = []

        for directory, entry in listings.items():
            try:
                content = entry[0] if isinstance(entry, tuple) else str(entry)
            except Exception:
                continue

            base = "" if directory in ("", ".") else directory.rstrip("/\\") + "/"

            for line in content.splitlines():
                line = line.strip()
                if line.startswith("[FILE] "):
                    name = line.replace("[FILE] ", "", 1).strip()
                    if name:
                        files.append((base + name).replace("\\", "/"))

        return files

    def _find_unique_file_by_ext(self, ext: str) -> Optional[str]:
        ext = ext.lower()
        matches = [
            f for f in self._known_files_from_dir_listings()
            if f.lower().endswith(ext)
        ]
        if len(matches) == 1:
            return matches[0]
        return None