"""
任务系统基础抽象。

完全对齐 Codex codex-rs/core/src/tasks/mod.rs：
- SessionTask trait  -> SessionTask ABC
- TaskKind           -> TaskKind 枚举
- TurnAbortReason    -> TurnAbortReason 枚举
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autocoder.tasks.scheduler import SessionTaskContext
    from autocoder.models.turn import TurnContext


class TaskKind(str, Enum):
    """对应 Codex 的 TaskKind"""
    REGULAR = "regular"      # 普通对话
    COMPACT = "compact"      # 历史压缩
    REVIEW = "review"        # 代码审查
    USER_SHELL = "user_shell"  # 用户直接执行 shell


class TurnAbortReason(str, Enum):
    """对应 Codex 的 TurnAbortReason"""
    INTERRUPTED = "interrupted"  # 用户主动打断 (Ctrl+C)
    REPLACED = "replaced"        # 被新任务替换
    ERROR = "error"              # 异常终止


class SessionTask(ABC):
    """
    对应 Codex 的 SessionTask trait。

    每个任务封装一种 Codex 工作流。任务由 TaskScheduler 拥有，
    在后台 asyncio.Task 中执行。
    """

    @abstractmethod
    def kind(self) -> TaskKind:
        """任务类型，用于遥测和 UI 展示。"""
        ...

    def span_name(self) -> str:
        """tracing span 名称。"""
        return f"session_task.{self.kind().value}"

    @abstractmethod
    async def run(
        self,
        session: "SessionTaskContext",
        ctx: "TurnContext",
        user_input: str,
        cancel_event: asyncio.Event,
    ) -> Optional[str]:
        """
        执行任务直到完成或被取消。

        对应 Codex SessionTask::run。
        - session: 任务运行所需的会话上下文
        - ctx: 本次 turn 的不可变上下文
        - user_input: 用户输入
        - cancel_event: 当 session 请求中断时被 set，实现要监听它并尽快退出

        返回 Some(message) 时，该消息会作为 turn 的最终回复发给用户。
        """
        ...

    async def abort(
        self,
        session: "SessionTaskContext",
        ctx: "TurnContext",
    ) -> None:
        """
        中断后的清理钩子。

        对应 Codex SessionTask::abort。默认无操作，
        需要额外清理资源时重写。
        """
        return None