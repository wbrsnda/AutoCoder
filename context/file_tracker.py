"""
文件上下文跟踪器。

借鉴 Codex context_manager 的设计：
- 跟踪已读文件，防止重复读取
- 维护文件摘要缓存，节省 token
- 为 Architect 提供上下文感知能力
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FileSnapshot:
    """单个文件的上下文快照"""
    path: str
    content_hash: str
    summary: str
    line_count: int
    read_at: float = field(default_factory=time.time)
    modified_at: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        """文件在读取后是否被修改过"""
        return self.modified_at is not None and self.modified_at > self.read_at


@dataclass
class ToolCallRecord:
    """单次工具调用记录，用于去重和审计"""
    tool_name: str
    args_hash: str
    result_preview: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True


class FileTracker:
    """
    文件上下文跟踪器。

    核心职责：
    1. 跟踪已读文件，防止 Coder 重复读同一文件
    2. 维护文件摘要缓存，为 Architect 提供上下文
    3. 跟踪已修改文件，标记需要重新读取的文件
    4. 跟踪工具调用历史，防止重复调用
    """

    def __init__(self):
        self._files: dict[str, FileSnapshot] = {}
        self._tool_history: list[ToolCallRecord] = []
        self._dir_listings: dict[str, tuple[str, float]] = {}  # dir -> (content, timestamp)

    # ── 文件跟踪 ──────────────────────────────────────────────

    def get_modified_files(self) -> list[str]:
        """返回被写入/patch 过的文件列表（删除不算 modified）"""
        return [p for p, snap in self._files.items() if snap.modified_at is not None]

    def record_file_read(
        self,
        file_path: str,
        content: str,
        summary: str = "",
    ) -> None:
        """记录一次文件读取"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        line_count = content.count("\n") + 1
        self._files[file_path] = FileSnapshot(
            path=file_path,
            content_hash=content_hash,
            summary=summary or f"{line_count} lines",
            line_count=line_count,
        )

    def record_file_modified(self, file_path: str) -> None:
        """记录文件被修改（写入/patch/删除）"""
        if file_path in self._files:
            self._files[file_path].modified_at = time.time()

    def record_file_deleted(self, file_path: str) -> None:
        """记录文件被删除"""
        self._files.pop(file_path, None)

    def is_file_read(self, file_path: str) -> bool:
        """检查文件是否已被读取且未过期"""
        snapshot = self._files.get(file_path)
        if snapshot is None:
            return False
        return not snapshot.is_stale

    def get_file_summary(self, file_path: str) -> Optional[str]:
        """获取已读文件的摘要"""
        snapshot = self._files.get(file_path)
        return snapshot.summary if snapshot else None

    # ── 目录跟踪 ─────────────────────────────────────────────

    def record_dir_listing(self, directory: str, content: str) -> None:
        """记录目录列表结果"""
        self._dir_listings[directory] = (content, time.time())

    def is_dir_listed(self, directory: str, max_age_seconds: float = 300) -> bool:
        """检查目录是否在最近 N 秒内已列出"""
        entry = self._dir_listings.get(directory)
        if entry is None:
            return False
        _, ts = entry
        return (time.time() - ts) < max_age_seconds

    # ── 工具调用去重 ──────────────────────────────────────────

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        result_preview: str = "",
        success: bool = True,
    ) -> None:
        """记录工具调用"""
        args_hash = hashlib.md5(
            f"{tool_name}:{sorted(tool_args.items())}".encode()
        ).hexdigest()[:12]
        self._tool_history.append(ToolCallRecord(
            tool_name=tool_name,
            args_hash=args_hash,
            result_preview=result_preview[:200],
            success=success,
        ))

    def is_duplicate_call(self, tool_name: str, tool_args: dict) -> bool:
        """检查是否是重复的工具调用"""
        args_hash = hashlib.md5(
            f"{tool_name}:{sorted(tool_args.items())}".encode()
        ).hexdigest()[:12]
        return any(
            r.args_hash == args_hash and r.tool_name == tool_name
            for r in self._tool_history[-20:]  # 只检查最近 20 次
        )

    # ── 上下文注入 ────────────────────────────────────────────

    def build_context_summary(self) -> str:
        """
        生成当前上下文摘要，注入到 Architect 的 System Prompt 中。

        这是防止重复读取的核心机制：
        Architect 看到已读文件列表后，就不会再委派 Coder 去读同一个文件。
        """
        if not self._files and not self._dir_listings:
            return ""

        parts = ["\n[CONTEXT TRACKER]"]

        if self._files:
            parts.append("Already read files (DO NOT re-read these):")
            for path, snap in self._files.items():
                status = "⚠️ MODIFIED since read" if snap.is_stale else "✓"
                parts.append(f"  - {path} [{status}] ({snap.summary})")

        if self._dir_listings:
            dirs = list(self._dir_listings.keys())
            parts.append(f"Already listed directories: {', '.join(dirs)}")

        parts.append(
            "If you need updated content of a MODIFIED file, "
            "delegate a re-read. Otherwise, do NOT re-read."
        )

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """返回跟踪统计信息"""
        return {
            "files_read": len(self._files),
            "files_stale": sum(1 for f in self._files.values() if f.is_stale),
            "dirs_listed": len(self._dir_listings),
            "tool_calls_total": len(self._tool_history),
        }