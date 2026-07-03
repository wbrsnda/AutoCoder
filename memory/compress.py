from langchain_core.messages import SystemMessage, ToolMessage
from autocoder.utils.config import Config

TOOL_MSG_MAX_CHARS = 600
SUMMARY_PREFIX = "[Session Summary]"


def is_summary_message(msg) -> bool:
    return isinstance(msg, SystemMessage) and str(msg.content).startswith(SUMMARY_PREFIX)


def truncate_tool_message(msg: ToolMessage) -> ToolMessage:
    content = str(msg.content)
    if len(content) <= TOOL_MSG_MAX_CHARS:
        return msg
    truncated = f"{content[:400]}\n...[Truncated {len(content)} chars]...\n{content[-150:]}"
    return ToolMessage(
        content=truncated,
        tool_call_id=msg.tool_call_id,
        name=getattr(msg, "name", None),
    )


async def compress_history(messages: list, llm, threshold: int = 24, keep_recent: int = 10) -> list:
    total_chars = sum(len(str(getattr(m, "content", "")) or "") for m in messages)
    
    # ★ 双重触发门槛：条数超过 threshold OR 字符总量超过 COMPRESS_MAX_CHARS
    need_compress = (len(messages) > threshold) or (total_chars > Config.COMPRESS_MAX_CHARS)
    print(f"🔎 [CompressCheck] msgs={len(messages)}/{threshold} chars={total_chars}/{Config.COMPRESS_MAX_CHARS} trigger={need_compress}")

    if not need_compress:
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
            content_str = str(m.content)
            if any(k in content_str for k in ["mcp_read_file", "class ", "def ", "[Already read]"]):
                filtered_old.append(m)
            else:
                filtered_old.append(truncate_tool_message(m))
        else:
            filtered_old.append(m)

    summary_prompt = SystemMessage(content=(
        "You are a session summarizer for a coding assistant.\n"
        "Summarize the conversation and preserve:\n"
        "1. The user's original goal\n"
        "2. Completed steps and every important tool call\n"
        "3. Files read / modified with exact paths\n"
        "4. Current progress and decisions\n"
        "5. Important findings for future continuation\n"
    ))

    try:
        print("🗜️ [Memory] Compressing old history...")
        res = await llm.ainvoke([summary_prompt] + filtered_old)
        compressed = SystemMessage(content=f"{SUMMARY_PREFIX}\n{str(res.content)}")
    except Exception as e:
        print(f"⚠️ [Memory] Compression failed: {e}")
        compressed = SystemMessage(content=f"{SUMMARY_PREFIX}\n[Previous {len(old_msgs)} messages summarized after failure]")

    result_msgs = sys_msgs + [compressed] + recent_msgs
    after_chars = sum(len(str(getattr(m, "content", "")) or "") for m in result_msgs)
    print(f"✅ [Memory] Compressed from {len(messages)} msgs ({total_chars} chars) to {len(result_msgs)} msgs ({after_chars} chars)")
    return result_msgs