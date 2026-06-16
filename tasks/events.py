"""
Turn 生命周期事件。

对应 Codex 的：
- TurnStartedEvent
- TurnCompleteEvent
- TurnAbortedEvent

Codex 通过 EventMsg 把这些事件流式推送给客户端。
我们先用回调 + 打印实现，未来可接入 SSE / WebSocket。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from autocoder.tasks.base import TurnAbortReason


@dataclass
class TurnStartedEvent:
    turn_id: str
    started_at: float


@dataclass
class TurnCompleteEvent:
    turn_id: str
    last_agent_message: Optional[str]
    duration_ms: Optional[int]


@dataclass
class TurnAbortedEvent:
    turn_id: str
    reason: TurnAbortReason
    duration_ms: Optional[int]


# 事件回调类型
EventHandler = Callable[[object], None]


class EventBus:
    """
    极简事件总线。

    对应 Codex Session::send_event 的角色。
    现阶段只做打印 + 可选回调，未来可替换为真正的流式通道。
    """

    def __init__(self, handler: Optional[EventHandler] = None):
        self._handler = handler

    def emit(self, event: object) -> None:
        if isinstance(event, TurnStartedEvent):
            print(f"🟢 [TurnStarted] {event.turn_id}")
        elif isinstance(event, TurnCompleteEvent):
            ms = event.duration_ms
            print(f"✅ [TurnComplete] {event.turn_id} | {ms}ms")
        elif isinstance(event, TurnAbortedEvent):
            print(f"🛑 [TurnAborted] {event.turn_id} | reason={event.reason.value} | {event.duration_ms}ms")

        if self._handler:
            self._handler(event)