"""
RegularTask：普通对话任务。

对应 Codex codex-rs/core/src/tasks/regular.rs。

Codex RegularTask::run 的核心是：
    loop {
        last_message = run_turn(...).await;
        if !has_pending_input { return last_message; }
    }

我们把 run_turn 替换为你已有的 graph.astream()，
并保留 pending input 循环（steering）。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

from autocoder.models.turn import TurnContext
from autocoder.tasks.base import SessionTask, TaskKind
from autocoder.tasks.scheduler import SessionTaskContext
from dataclasses import asdict

class RegularTask(SessionTask):
    """对应 Codex RegularTask。"""

    def kind(self) -> TaskKind:
        return TaskKind.REGULAR

    async def run(
        self,
        session: SessionTaskContext,
        ctx: TurnContext,
        user_input: str,
        cancel_event: asyncio.Event,
    ) -> Optional[str]:
        next_input = user_input
        last_message: Optional[str] = None

        # 对应 Codex 的 loop { run_turn() }
        while True:
            last_message = await self._run_one_turn(
                session, ctx, next_input, cancel_event
            )

            if cancel_event.is_set():
                return last_message

            # 处理 pending input（steering）
            # 这里需要 scheduler 引用，简化处理：由外层 scheduler 控制
            # 本轮先不在 task 内部取 pending，交给 scheduler.on_finished
            return last_message

    async def _run_one_turn(
        self,
        session: SessionTaskContext,
        ctx: TurnContext,
        user_input: str,
        cancel_event: asyncio.Event,
    ) -> Optional[str]:
        messages = list(session.injected_messages)
        session.injected_messages = []
        messages.append(HumanMessage(content=user_input))

        # ★ 不再把 turn_ctx 塞进 state！
        state = {
            "messages": messages,
            "tool_call_count": 0,
            "current_role": "architect",
            "delegation": "",
            "budget_exhausted": False,
            "latest_tool_results": [],
            "guard_retries": 0,
        }

        last_message: Optional[str] = None

        async for step in session.graph.astream(state, config=session.lang_config):
            if cancel_event.is_set():
                print("🟡 [RegularTask] Cancel detected mid-turn, stopping astream")
                break

            for node_name, node_output in step.items():
                msgs = node_output.get("messages", []) if isinstance(node_output, dict) else []
                for m in msgs:
                    if isinstance(m, AIMessage) and m.content:
                        last_message = m.content

        return last_message