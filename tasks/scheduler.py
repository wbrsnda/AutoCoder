"""
任务调度器。

完全对齐 Codex codex-rs/core/src/tasks/mod.rs 的 Session impl：
- spawn_task          -> spawn_task
- start_task          -> _start_task
- abort_all_tasks     -> abort_all_tasks
- on_task_finished    -> _on_task_finished
- handle_task_abort   -> _handle_task_abort（含优雅中断 + 中断标记）

关键设计（来自 Codex）：
1. CancellationToken      -> asyncio.Event (cancel_event)
2. tokio::spawn           -> asyncio.create_task
3. GRACEFULL_INTERRUPTION -> 100ms grace period
4. interrupted marker     -> 中断后向历史注入提示
5. pending input          -> turn 运行中用户的新输入（steering）
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from langchain_core.messages import HumanMessage

from autocoder.models.turn import TurnContext, TurnStatus
from autocoder.tasks.base import SessionTask, TurnAbortReason
from autocoder.tasks.events import (
    EventBus,
    TurnStartedEvent,
    TurnCompleteEvent,
    TurnAbortedEvent,
)

# 对应 Codex GRACEFULL_INTERRUPTION_TIMEOUT_MS = 100
GRACEFUL_INTERRUPTION_TIMEOUT_S = 0.1

# 对应 Codex 的中断标记文案（context/turn_aborted.rs）
INTERRUPTED_GUIDANCE = (
    "[The previous turn was interrupted by the user. "
    "The user may provide new instructions. "
    "Do not assume the interrupted work was completed.]"
)


class SessionTaskContext:
    """
    对应 Codex SessionTaskContext。

    暴露任务运行所需的会话片段。这里我们持有：
    - graph: 你已有的 LangGraph
    - config: LangGraph 的 thread 配置
    - history: 共享对话历史（用于注入中断标记）
    """

    def __init__(self, graph, lang_config: dict, event_bus: EventBus):
        self.graph = graph
        self.lang_config = lang_config
        self.event_bus = event_bus
        # 共享历史，用于中断标记注入；与 LangGraph checkpointer 协作
        self.injected_messages: list = []


class RunningTask:
    """
    对应 Codex 的 RunningTask。

    持有正在运行的 asyncio.Task、取消信号、完成通知、turn 上下文。
    """

    def __init__(
        self,
        task: SessionTask,
        ctx: TurnContext,
        handle: asyncio.Task,
        cancel_event: asyncio.Event,
        done: asyncio.Event,
    ):
        self.task = task
        self.ctx = ctx
        self.handle = handle
        self.cancel_event = cancel_event
        self.done = done


class TaskScheduler:
    """
    任务调度器，对应 Codex Session 中的任务管理部分。

    一次只允许一个 active task（与 Codex 的 active_turn 语义一致）。
    """

    def __init__(self, session_ctx: SessionTaskContext):
        self.session_ctx = session_ctx
        self._active: Optional[RunningTask] = None
        # pending input：turn 运行中用户输入的新内容（steering）
        self._pending_input: list[str] = []
        self._lock = asyncio.Lock()

    # ── 对应 Codex spawn_task ──────────────────────────────────
    async def spawn_task(
        self,
        task: SessionTask,
        ctx: TurnContext,
        user_input: str,
    ) -> None:
        """
        启动一个新任务。

        对应 Codex spawn_task：先 abort 旧任务（Replaced），再启动新任务。
        """
        await self.abort_all_tasks(TurnAbortReason.REPLACED)
        await self._start_task(task, ctx, user_input)

    # ── 对应 Codex start_task ──────────────────────────────────
    async def _start_task(
        self,
        task: SessionTask,
        ctx: TurnContext,
        user_input: str,
    ) -> None:
        ctx.status = TurnStatus.RUNNING
        ctx.started_at = time.time()

        cancel_event = asyncio.Event()
        done = asyncio.Event()

        # emit TurnStarted（对应 Codex emit_turn_start_lifecycle）
        self.session_ctx.event_bus.emit(
            TurnStartedEvent(turn_id=ctx.turn_id, started_at=ctx.started_at)
        )

        async def _runner():
            last_message: Optional[str] = None
            try:
                last_message = await task.run(
                    self.session_ctx, ctx, user_input, cancel_event
                )
            except asyncio.CancelledError:
                # 被强制取消，交给 abort 流程处理
                raise
            except Exception as e:
                ctx.status = TurnStatus.FAILED
                ctx.aborted_reason = str(e)
                print(f"❌ [Task] Run failed: {e}")
            finally:
                done.set()

            # 正常完成（未被取消）才走 on_task_finished
            if not cancel_event.is_set():
                await self._on_task_finished(ctx, last_message)

        handle = asyncio.create_task(_runner())
        self._active = RunningTask(task, ctx, handle, cancel_event, done)

    # ── 对应 Codex on_task_finished ────────────────────────────
    async def _on_task_finished(
        self,
        ctx: TurnContext,
        last_agent_message: Optional[str],
    ) -> None:
        ctx.complete(last_agent_message)

        # emit TurnComplete（对应 Codex EventMsg::TurnComplete）
        self.session_ctx.event_bus.emit(
            TurnCompleteEvent(
                turn_id=ctx.turn_id,
                last_agent_message=last_agent_message,
                duration_ms=ctx.duration_ms,
            )
        )

        # 处理 pending input（对应 Codex pending_input / steering）
        if self._pending_input:
            print(f"📨 [Scheduler] {len(self._pending_input)} pending input(s) queued")

        self._active = None

    # ── 对应 Codex abort_all_tasks ─────────────────────────────
    async def abort_all_tasks(self, reason: TurnAbortReason) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        await self._handle_task_abort(active, reason)

    # ── 对应 Codex handle_task_abort（优雅中断 + 中断标记）──────
    async def _handle_task_abort(
        self,
        running: RunningTask,
        reason: TurnAbortReason,
    ) -> None:
        if running.cancel_event.is_set():
            return

        print(f"🟠 [Scheduler] Aborting task {running.ctx.turn_id} (reason={reason.value})")

        # 1. 发出取消信号（对应 cancellation_token.cancel()）
        running.cancel_event.set()

        # 2. 等待 grace period（对应 GRACEFULL_INTERRUPTION_TIMEOUT_MS）
        try:
            await asyncio.wait_for(
                running.done.wait(),
                timeout=GRACEFUL_INTERRUPTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            print(f"⚠️  [Scheduler] Task didn't stop gracefully in "
                  f"{GRACEFUL_INTERRUPTION_TIMEOUT_S*1000:.0f}ms, forcing abort")

        # 3. 强制取消（对应 handle.abort()）
        if not running.handle.done():
            running.handle.cancel()
            try:
                await running.handle
            except asyncio.CancelledError:
                pass

        # 4. 任务自定义清理（对应 session_task.abort()）
        await running.task.abort(self.session_ctx, running.ctx)

        # 5. 中断标记注入（对应 interrupted_turn_history_marker）
        if reason == TurnAbortReason.INTERRUPTED:
            self.session_ctx.injected_messages.append(
                HumanMessage(content=INTERRUPTED_GUIDANCE, name="SystemRuntime")
            )
            print("📌 [Scheduler] Injected interrupt marker into history")

        # 6. 更新状态 + emit TurnAborted
        running.ctx.interrupt(reason.value)
        self.session_ctx.event_bus.emit(
            TurnAbortedEvent(
                turn_id=running.ctx.turn_id,
                reason=reason,
                duration_ms=running.ctx.duration_ms,
            )
        )

    # ── pending input（steering）───────────────────────────────
    def queue_pending_input(self, text: str) -> None:
        """turn 运行中用户输入新内容时调用。"""
        self._pending_input.append(text)

    def take_pending_input(self) -> list[str]:
        items = self._pending_input
        self._pending_input = []
        return items

    @property
    def is_active(self) -> bool:
        return self._active is not None
    