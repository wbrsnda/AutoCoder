import asyncio
import operator
import platform
from typing import Annotated, Literal, TypedDict
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from autocoder.agent.prompts import ARCHITECT_SYSTEM, CODER_SYSTEM
from autocoder.memory.compress import compress_history
from autocoder.orchestrator.hook_engine import HookEngine, HookEvent
from autocoder.skills.builtin import match_skill

OS_NAME = platform.system()
ROLE_ARCHITECT = "architect"
ROLE_CODER = "coder"


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_call_count: int
    current_role: str
    delegation: str
    budget_exhausted: bool


def build_graph(
    architect_llm,
    coder_llm,
    mcp_tools: list,
    orchestrator,
    hook_engine: HookEngine = None,
    rag_tool=None,
    max_tool_calls_per_turn: int = 15,
    workspace_dir: str = ".",
):
    if hook_engine is None:
        hook_engine = HookEngine()

    all_tools = mcp_tools + ([rag_tool] if rag_tool else [])
    architect_bound = architect_llm
    coder_bound = coder_llm.bind_tools(all_tools)
    tool_map = {t.name: t for t in all_tools}

    # ── Architect 节点 ────────────────────────────────────────
    async def architect_node(state: AgentState) -> dict:
        messages = await compress_history(state["messages"], architect_llm)

        last_user = next(
            (m.content for m in reversed(messages)
             if isinstance(m, HumanMessage) and getattr(m, "name", "") != "SystemRuntime"),
            ""
        )
        skill = match_skill(last_user)

        project_ctx = f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        sys_content = ARCHITECT_SYSTEM.replace("__PROJECT_CONTEXT__", project_ctx)
        if skill:
            sys_content += "\n\n[SKILL ACTIVATED]:\n" + "\n".join(skill["steps"])

        clean = [m for m in messages if not isinstance(m, SystemMessage)]
        msgs = [SystemMessage(content=sys_content)] + clean

        res = await architect_bound.ainvoke(msgs)
        content = (res.content or "").strip()

        if not content or content == "None":
            print("\n⚠️  [Guard] Architect empty. Injecting guidance...")
            guidance = HumanMessage(
                content="[System]: You must output either a direct answer ending with AWAITING USER INPUT or DELEGATE TO CODER.",
                name="SystemRuntime"
            )
            return {
                "messages": [res, guidance],
                "current_role": ROLE_ARCHITECT,
                "delegation": "",
                "tool_call_count": 0,
                "budget_exhausted": False,
            }

        print(f"\n🏛️  Architect:\n{content}\n")

        delegation = ""
        if "DELEGATE TO CODER:" in content:
            raw = content.split("DELEGATE TO CODER:", 1)[-1].strip()
            delegation = raw.split("AWAITING USER INPUT")[0].strip()

        return {
            "messages": [res],
            "current_role": ROLE_ARCHITECT,
            "delegation": delegation,
            "tool_call_count": 0,
            "budget_exhausted": False,
        }

    # ── Coder 节点 ────────────────────────────────────────────
    async def coder_node(state: AgentState) -> dict:
        messages = await compress_history(state["messages"], architect_llm)
        delegation = state.get("delegation", "")

        sys_content = CODER_SYSTEM + f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        if delegation:
            sys_content += f"\n\n[CURRENT TASK FROM ARCHITECT]: {delegation}"

        clean = [m for m in messages if not isinstance(m, SystemMessage)]
        msgs = [SystemMessage(content=sys_content)] + clean

        res = await coder_bound.ainvoke(msgs)
        content = (res.content or "").strip()

        if content:
            print(f"\n💻 Coder:\n{content}\n")
        else:
            print("\n🔧 Coder: thinking...")

        if res.tool_calls:
            current_names = [tc["name"] for tc in res.tool_calls]

            # 统计最近的 mcp_read_file 调用次数
            read_file_count = 0
            for m in reversed(messages[-10:]):
                if hasattr(m, "tool_calls") and m.tool_calls:
                    if any(tc["name"] == "mcp_read_file" for tc in m.tool_calls):
                        read_file_count += 1
                    if read_file_count >= 2:
                        break

            # 如果连续多次调用 mcp_read_file，则强制报告
            if "mcp_read_file" in current_names and read_file_count >= 2:
                print("⚠️  [Guard] Repeated mcp_read_file calls detected. Forcing report.")
                inject = HumanMessage(
                    content="[System]: You have already called mcp_read_file multiple times. Stop calling tools and output REPORT TO ARCHITECT with what you have.",
                    name="SystemRuntime"
                )
                return {"messages": [res, inject], "current_role": ROLE_CODER}

            print(f"🛠️  Tools: {current_names}")

        return {"messages": [res], "current_role": ROLE_CODER}

    # ── hooked_tools_node ─────────────────────────────────────
    async def hooked_tools_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not getattr(last_msg, "tool_calls", None):
            return {"messages": []}

        async def execute_single(tool_call):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            call_id = tool_call["id"]

            hook_ctx = {"tool_name": tool_name, **tool_args}
            hook_result = hook_engine.evaluate(HookEvent.PRE_TOOL_USE, tool_name=tool_name, context=hook_ctx)

            if hook_result.should_block:
                return ToolMessage(content=f"Blocked: {hook_result.block_reason}", tool_call_id=call_id)

            if tool_name not in tool_map:
                return ToolMessage(content="Tool not found", tool_call_id=call_id)

            try:
                tool_obj = tool_map[tool_name]
                result = await tool_obj.ainvoke(tool_args) if hasattr(tool_obj, "ainvoke") else tool_obj.invoke(tool_args)
                result_str = str(result)
            except Exception as e:
                result_str = f"Error: {e}"

            hook_engine.evaluate(HookEvent.POST_TOOL_USE, tool_name=tool_name, context={**hook_ctx, "result": result_str[:300]})
            return ToolMessage(content=result_str, tool_call_id=call_id)

        tool_messages = await asyncio.gather(*[execute_single(tc) for tc in last_msg.tool_calls])
        return {"messages": list(tool_messages)}

    # ── Budget 节点 ───────────────────────────────────────────
    async def budget_node(state: AgentState) -> dict:
        count = state.get("tool_call_count", 0) + 1
        if count >= max_tool_calls_per_turn:
            steering = HumanMessage(
                content="[System]: Budget exhausted. Output REPORT TO ARCHITECT.",
                name="SystemRuntime"
            )
            return {"messages": [steering], "tool_call_count": count, "budget_exhausted": True}
        return {"tool_call_count": count}

    def route_after_architect(state: AgentState):
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", "") or ""
        delegation = state.get("delegation", "").strip()

        if getattr(last_msg, "name", "") == "SystemRuntime":
            return "architect"
        if "AWAITING USER INPUT" in content:
            return END
        return "coder" if delegation else END


    def route_after_coder(state: AgentState):
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", "") or ""

        # 有工具调用 → 去执行工具
        if getattr(last_msg, "tool_calls", None):
            return "hooked_tools"

        # SystemRuntime 消息允许自循环
        if getattr(last_msg, "name", "") == "SystemRuntime":
            return "coder"

        # 明确要求汇报或预算耗尽 → 回到 Architect
        if state.get("budget_exhausted", False) or "REPORT TO ARCHITECT" in content:
            return "architect"

        # 默认情况：Coder 已给出最终回复，结束 turn 并返回用户
        return END


    def route_after_budget(state: AgentState):
        return "coder"

    # ── 构建图 ────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("architect", architect_node)
    builder.add_node("coder", coder_node)
    builder.add_node("hooked_tools", hooked_tools_node)
    builder.add_node("budget", budget_node)

    builder.set_entry_point("architect")

    builder.add_conditional_edges(
        "architect",
        route_after_architect,
        {"coder": "coder", "architect": "architect", END: END}
    )
    builder.add_conditional_edges(
        "coder",
        route_after_coder,
        {
            "hooked_tools": "hooked_tools",
            "architect": "architect",
            "coder": "coder",      # ← 关键：允许自循环
            END: END,
        }
    )
    builder.add_edge("hooked_tools", "budget")
    builder.add_conditional_edges("budget", route_after_budget, {"coder": "coder"})

    return builder.compile(checkpointer=MemorySaver())