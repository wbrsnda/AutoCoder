"""
文件上下文跟踪器。

借鉴 Codex context_manager 的设计：
- 跟踪已读文件，防止重复读取
- 维护文件摘要缓存，节省 token
- 跟踪 write / append / patch / delete，标记文件是否过期
- 为 Architect 提供上下文感知能力
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FileSnapshot:
    """单个文件的上下文快照"""
    path: str
    content_hash: str
    summary: str
    line_count: int

    # 是否真的读过文件内容
    was_read: bool = True

    read_at: float = field(default_factory=time.time)
    modified_at: Optional[float] = None
    appended_at: Optional[float] = None
    deleted_at: Optional[float] = None

    @property
    def is_stale(self) -> bool:
        """文件在读取后是否被修改/追加/删除过"""
        if not self.was_read:
            return False

        if self.modified_at is not None and self.modified_at > self.read_at:
            return True
        if self.appended_at is not None and self.appended_at > self.read_at:
            return True
        if self.deleted_at is not None and self.deleted_at > self.read_at:
            return True
        return False

    @property
    def change_type(self) -> str:
        """返回相对 read_at 的变更类型"""
        if not self.was_read:
            if self.deleted_at is not None:
                return "DELETED"
            if self.modified_at is not None:
                return "MODIFIED"
            if self.appended_at is not None:
                return "APPENDED"
            return "UNREAD"

        candidates: list[tuple[str, float]] = []
        if self.modified_at is not None and self.modified_at > self.read_at:
            candidates.append(("MODIFIED", self.modified_at))
        if self.appended_at is not None and self.appended_at > self.read_at:
            candidates.append(("APPENDED", self.appended_at))
        if self.deleted_at is not None and self.deleted_at > self.read_at:
            candidates.append(("DELETED", self.deleted_at))

        if not candidates:
            return "UNCHANGED"

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


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
    3. 跟踪已修改/追加文件，标记需要重新读取的文件
    4. 跟踪工具调用历史，防止重复调用
    """

    def __init__(self):
        self._files: dict[str, FileSnapshot] = {}
        self._tool_history: list[ToolCallRecord] = []
        self._dir_listings: dict[str, tuple[str, float]] = {}

    # ── 内部工具 ──────────────────────────────────────────────

    @staticmethod
    def _hash_tool_args(tool_name: str, tool_args: dict) -> str:
        return hashlib.md5(
            f"{tool_name}:{sorted(tool_args.items())}".encode()
        ).hexdigest()[:12]

    def _ensure_unread_snapshot(self, file_path: str, summary: str = "") -> FileSnapshot:
        """
        为未读但发生变更的文件创建一个 tracking snapshot。
        注意：was_read=False，所以不会被当成“已读文件”。
        """
        snap = self._files.get(file_path)
        if snap is None:
            snap = FileSnapshot(
                path=file_path,
                content_hash="",
                summary=summary or "Changed file (not read in this session)",
                line_count=0,
                was_read=False,
                read_at=0.0,
            )
            self._files[file_path] = snap
        return snap

    # ── 文件跟踪 ──────────────────────────────────────────────

    def get_modified_files(self) -> list[str]:
        """返回被 write / append / patch 过的文件列表"""
        return [
            p for p, snap in self._files.items()
            if snap.modified_at is not None or snap.appended_at is not None
        ]

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
            was_read=True,
            read_at=time.time(),
        )

    def record_file_modified(self, file_path: str) -> None:
        """记录文件被覆盖写入或 patch 修改"""
        now = time.time()
        if file_path in self._files:
            self._files[file_path].modified_at = now
        else:
            snap = self._ensure_unread_snapshot(
                file_path,
                summary="Modified file (not read in this session)",
            )
            snap.modified_at = now

    def record_file_appended(self, file_path: str) -> None:
        """记录文件被追加"""
        now = time.time()
        if file_path in self._files:
            self._files[file_path].appended_at = now
        else:
            snap = self._ensure_unread_snapshot(
                file_path,
                summary="Appended file (not read in this session)",
            )
            snap.appended_at = now

    def record_file_deleted(self, file_path: str) -> None:
        """记录文件被删除"""
        now = time.time()
        if file_path in self._files:
            self._files[file_path].deleted_at = now
        else:
            snap = self._ensure_unread_snapshot(
                file_path,
                summary="Deleted file (not read in this session)",
            )
            snap.deleted_at = now

    def is_file_read(self, file_path: str) -> bool:
        """检查文件是否已被读取且未过期"""
        snapshot = self._files.get(file_path)
        if snapshot is None:
            return False
        return snapshot.was_read and not snapshot.is_stale

    def get_file_summary(self, file_path: str) -> Optional[str]:
        """获取已读文件的摘要"""
        snapshot = self._files.get(file_path)
        if snapshot and snapshot.was_read:
            return snapshot.summary
        return None

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
        args_hash = self._hash_tool_args(tool_name, tool_args)
        self._tool_history.append(ToolCallRecord(
            tool_name=tool_name,
            args_hash=args_hash,
            result_preview=result_preview[:200],
            success=success,
        ))

    def is_duplicate_call(self, tool_name: str, tool_args: dict) -> bool:
        """检查是否是重复的工具调用"""
        args_hash = self._hash_tool_args(tool_name, tool_args)
        return any(
            r.args_hash == args_hash and r.tool_name == tool_name
            for r in self._tool_history[-20:]
        )

    # ── 上下文注入 ────────────────────────────────────────────

    def build_context_summary(self) -> str:
        """
        生成当前上下文摘要，注入到 Architect / Coder 的 System Prompt 中。

        修复点：
        - 不再只写 "Already listed directories: ."
        - 会带上目录缓存中的文件名，避免模型忘记 project_experience.tex 这种文件。
        """
        if not self._files and not self._dir_listings:
            return ""

        parts = ["\n[CONTEXT TRACKER]"]

        read_files = {
            path: snap for path, snap in self._files.items()
            if snap.was_read and snap.deleted_at is None
        }
        changed_unread_files = {
            path: snap for path, snap in self._files.items()
            if not snap.was_read
            and (
                snap.modified_at is not None
                or snap.appended_at is not None
                or snap.deleted_at is not None
            )
        }

        if read_files:
            parts.append("Already read files:")
            for path, snap in read_files.items():
                if snap.is_stale:
                    status = f"⚠️ {snap.change_type} since read"
                else:
                    status = "✓"
                parts.append(f"  - {path} [{status}] ({snap.summary})")

        if changed_unread_files:
            parts.append("Changed files not read in this session:")
            for path, snap in changed_unread_files.items():
                parts.append(f"  - {path} [⚠️ {snap.change_type}] ({snap.summary})")

        # ★ 增强：注入已列目录的实际文件名
        if self._dir_listings:
            parts.append("Already listed directories and visible entries:")
            for directory, (content, ts) in self._dir_listings.items():
                files, dirs = [], []
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("[FILE] "):
                        files.append(line.replace("[FILE] ", "", 1))
                    elif line.startswith("[DIR] "):
                        dirs.append(line.replace("[DIR] ", "", 1))

                dir_label = directory or "."
                parts.append(f"  - {dir_label}:")
                if dirs:
                    parts.append(f"    dirs: {', '.join(dirs[:20])}")
                if files:
                    parts.append(f"    files: {', '.join(files[:40])}")
                    if len(files) > 40:
                        parts.append(f"    ... (+{len(files) - 40} more files)")
                if not dirs and not files:
                    parts.append("    (empty or no visible entries)")

        parts.append(
            "If you need updated content of a MODIFIED/APPENDED file, "
            "delegate a re-read. Otherwise, do NOT re-read already read unchanged files."
        )

        return "\n".join(parts)

    def get_stats(self) -> dict:
        """返回跟踪统计信息"""
        files_read = sum(1 for f in self._files.values() if f.was_read and f.deleted_at is None)
        files_stale = sum(1 for f in self._files.values() if f.is_stale)
        changed_unread = sum(
            1 for f in self._files.values()
            if not f.was_read and (
                f.modified_at is not None
                or f.appended_at is not None
                or f.deleted_at is not None
            )
        )

        return {
            "files_read": files_read,
            "files_stale": files_stale,
            "changed_unread": changed_unread,
            "dirs_listed": len(self._dir_listings),
            "tool_calls_total": len(self._tool_history),
        }