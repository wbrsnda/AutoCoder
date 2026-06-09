"""
历史压缩模块（单版本，修复重复定义 bug）。
"""
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

TOOL_MSG_MAX_CHARS = 500
SUMMARY_PREFIX = "[Session Summary]"


def is_summary_message(msg) -> bool:
    if isinstance(msg, SystemMessage):
        content = msg.content if isinstance(msg.content, str) else ""
        return content.startswith(SUMMARY_PREFIX)
    return False


def truncate_tool_message(msg: ToolMessage) -> ToolMessage:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    if len(content) <= TOOL_MSG_MAX_CHARS:
        return msg
    head = content[:400]
    tail = content[-100:]
    truncated = f"{head}\n...[Truncated {len(content)} chars]...\n{tail}"
    return ToolMessage(content=truncated, tool_call_id=msg.tool_call_id)


async def compress_history(
    messages: list,
    llm,
    threshold: int = 30,
    keep_recent: int = 10,
) -> list:
    if len(messages) <= threshold:
        return messages

    sys_msgs = [m for m in messages if isinstance(m, SystemMessage) and not is_summary_message(m)]
    non_sys = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_sys) <= keep_recent:
        return messages

    old_msgs = non_sys[:-keep_recent]
    recent_msgs = non_sys[-keep_recent:]

    if not old_msgs:
        return messages

    filtered_old = []
    for m in old_msgs:
        if isinstance(m, ToolMessage):
            # 只截断非 read_file 的工具消息
            # read_file 的内容是 Coder 分析的核心依据，不能截断
            if "mcp_read_file" in str(m.tool_call_id) or "file_path" in str(m.content)[:50]:
                filtered_old.append(m)
            else:
                filtered_old.append(truncate_tool_message(m))
        else:
            filtered_old.append(m)

    summary_prompt = SystemMessage(content=(
        "You are a session summarizer. Summarize the following conversation.\n"
        "You MUST include:\n"
        "1. The user's original goal\n"
        "2. COMPLETED STEPS: List every tool call that was made and what it returned "
        "(especially file reads - list EVERY filename that was already read)\n"
        "3. Files created or modified (with exact paths)\n"
        "4. Current progress status: what is done, what remains\n"
        "5. Key findings so far\n\n"
        "Format the 'COMPLETED STEPS' section clearly so the agent knows "
        "exactly which files have already been read and does not re-read them."
    ))

    try:
        print("🗜️  [Memory] Compressing old history...")
        response = await llm.ainvoke([summary_prompt] + filtered_old)
        summary_text = response.content if isinstance(response.content, str) else str(response.content)
        compressed = SystemMessage(content=f"{SUMMARY_PREFIX}\n{summary_text}")
        print(f"🗜️  [Memory] Compressed {len(old_msgs)} messages → 1 summary")
    except Exception as e:
        print(f"⚠️  [Memory] Compression failed ({e}), truncating")
        compressed = SystemMessage(content=f"{SUMMARY_PREFIX}\n[Previous {len(old_msgs)} messages truncated]")

    return sys_msgs + [compressed] + recent_msgs