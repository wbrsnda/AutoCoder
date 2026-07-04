"""
Token 感知的历史压缩 - 对齐 Codex run_pre_sampling_compact + replace_compacted_history

关键：
1. 只在 token 超阈值时触发（不再无脑数字符）
2. 生成 summary 后，使用 RemoveMessage 真正从 LangGraph state 移除旧消息
3. 记录压缩过程本身的 API usage
"""
from __future__ import annotations
from typing import Optional

from langchain_core.messages import (
    SystemMessage, ToolMessage, AIMessage, HumanMessage, RemoveMessage
)

from autocoder.context.token_tracker import (
    TokenTracker, truncate_text_by_tokens, estimate_message_tokens
)

SUMMARY_PREFIX = "[Session Summary]"
TOOL_MSG_TOKEN_CAP = 400  # 压缩时保留的工具输出上限


def is_summary_message(msg) -> bool:
    return isinstance(msg, SystemMessage) and str(msg.content).startswith(SUMMARY_PREFIX)


def _truncate_tool_for_summary(msg: ToolMessage) -> ToolMessage:
    """压缩前先把超长工具输出裁短，减少 summary LLM 调用成本"""
    content = str(msg.content)
    truncated, was_cut = truncate_text_by_tokens(content, TOOL_MSG_TOKEN_CAP)
    if not was_cut:
        return msg
    return ToolMessage(
        content=truncated,
        tool_call_id=msg.tool_call_id,
        name=getattr(msg, "name", None),
    )


SUMMARY_SYSTEM_PROMPT = (
    "You are a session summarizer for a coding assistant.\n"
    "Produce a concise but complete summary preserving:\n"
    "1. The user's original goal\n"
    "2. Every important tool call and its outcome (success/failure)\n"
    "3. Files read / modified with exact paths\n"
    "4. Key decisions and current progress\n"
    "5. Any errors encountered and their causes\n"
    "Format as structured markdown. Keep under 800 tokens."
)


async def compress_history_if_needed(
    messages: list,
    llm,
    token_tracker: TokenTracker,
    keep_recent: int = 10,
) -> tuple[list, list, bool]:
    """
    核心压缩函数。

    返回:
      (updates_for_state, prompt_messages, was_compressed)
      - updates_for_state: 要写回 LangGraph state 的 message 增量（含 RemoveMessage）
      - prompt_messages:   本次 ainvoke 实际用的 messages（已压缩）
      - was_compressed:    是否真的压缩了

    如果没触发压缩：updates_for_state=[], prompt_messages=messages
    """
    if not token_tracker.should_compact(messages):
        return [], messages, False

    hard = token_tracker.is_hard_limit_reached(messages)
    print(f"{'🔴' if hard else '🟡'} [Compact] Trigger "
          f"({'HARD LIMIT' if hard else 'soft threshold'})")

    # ── 划分 keep / drop ──
    sys_msgs = [m for m in messages if isinstance(m, SystemMessage) and not is_summary_message(m)]
    summary_msgs = [m for m in messages if is_summary_message(m)]
    non_sys = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_sys) <= keep_recent:
        return [], messages, False

    old = non_sys[:-keep_recent]
    recent = non_sys[-keep_recent:]

    # ── 准备用于 summary 的输入（工具消息裁短）──
    old_for_summary = []
    for m in old:
        if isinstance(m, ToolMessage):
            old_for_summary.append(_truncate_tool_for_summary(m))
        else:
            old_for_summary.append(m)

    # ── 调 LLM 生成 summary ──
    summary_text: str
    try:
        print("🗜️ [Compact] Generating summary...")
        res = await llm.ainvoke([
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            *old_for_summary,
        ])
        summary_text = str(res.content).strip()
        # 压缩本身也消耗 token，记录进 tracker
        token_tracker.record_ai_message(res)
    except Exception as e:
        print(f"⚠️ [Compact] Summary failed: {e}")
        summary_text = f"[Auto-summary of {len(old)} previous messages failed: {e}]"

    compressed_summary = SystemMessage(
        content=f"{SUMMARY_PREFIX}\n{summary_text}"
    )

    # ── 组装 LangGraph state 更新 ──
    # 用 RemoveMessage 干掉旧的 (包括旧 summary)，追加新 summary
    updates: list = []
    for m in old:
        mid = getattr(m, "id", None)
        if mid:
            updates.append(RemoveMessage(id=mid))
    for m in summary_msgs:
        mid = getattr(m, "id", None)
        if mid:
            updates.append(RemoveMessage(id=mid))
    updates.append(compressed_summary)

    # ── prompt_messages: 本次 ainvoke 实际用 ──
    prompt_messages = sys_msgs + [compressed_summary] + recent

    before_total = token_tracker.estimate_total(messages)
    after_total = token_tracker.estimate_total(prompt_messages)
    print(f"✅ [Compact] {before_total:,} → {after_total:,} tokens "
          f"({len(messages)} → {len(prompt_messages)} msgs)")

    return updates, prompt_messages, True