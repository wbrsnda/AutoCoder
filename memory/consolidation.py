from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

async def compress_history(messages: list, llm, threshold: int = 20) -> list:
    if len(messages) <= threshold:
        return messages

    sys_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    old_msgs = messages[1:-10] if len(messages) > 11 else messages[1:]
    recent_msgs = messages[-10:]

    if not old_msgs:
        return messages

    # 生成包含工具调用结果的摘要
    summary_prompt = SystemMessage(content=(
        "Summarize the conversation and tool calls into a concise report. "
        "Include: (1) Project goal, (2) Files created/modified, (3) Tool calls and their outcomes (success/failure), "
        "(4) Current project state. Be specific."
    ))
    try:
        response = await llm.ainvoke([summary_prompt] + old_msgs)
        compressed = SystemMessage(content=f"[Session Summary]:\n{response.content}")
    except Exception:
        compressed = SystemMessage(content="[Previous session compressed due to length]")

    rebuilt = [compressed]
    if sys_msg:
        rebuilt.insert(0, sys_msg)
    rebuilt.extend(recent_msgs)
    return rebuilt