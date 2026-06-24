import asyncio
import operator
import platform
import ast
import re
from typing import Annotated, TypedDict, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from autocoder.agent.prompts import ARCHITECT_SYSTEM, CODER_SYSTEM
from autocoder.memory.compress import compress_history
from autocoder.orchestrator.hook_engine import HookEngine, HookEvent
from autocoder.skills.builtin import match_skill
from autocoder.context.file_tracker import FileTracker
from autocoder.models.turn import TurnContext, TurnStatus, TurnPhase

OS_NAME = platform.system()
ROLE_ARCHITECT = "architect"
ROLE_CODER = "coder"
ROLE_CODER_REPORT = "coder_report"


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_call_count: int
    current_role: str
    delegation: str
    budget_exhausted: bool
    latest_tool_results: list


def _strip_line_numbers(text: str) -> str:
    cleaned = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+\s*\|\s?(.*)$", line)
        cleaned.append(match.group(1) if match else line)
    return "\n".join(cleaned)


def _summarize_list_dir(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    files, dirs = [], []
    for line in lines:
        if line.startswith("[FILE] "):
            files.append(line.replace("[FILE] ", "", 1))
        elif line.startswith("[DIR] "):
            dirs.append(line.replace("[DIR] ", "", 1))
    parts = ["### 目录内容"]
    if dirs:
        parts.append(f"- 目录: {', '.join(dirs)}")
    if files:
        parts.append(f"- 文件: {', '.join(files)}")
    if not dirs and not files:
        parts.append("- 目录为空或没有可见文件。")
    return "\n".join(parts)


def _collect_code_patterns(source: str) -> dict:
    lines = source.splitlines()
    patterns = {
        "数据加载": ["Dataset", "DataLoader", "read_csv", "torch.load", "np.load"],
        "模型定义": ["nn.Module", "TransformerEncoder", "Linear(", "LayerNorm"],
        "训练循环": ["model.train()", "optimizer.zero_grad()", "loss.backward()"],
        "验证评估": ["model.eval()", "torch.no_grad()", "evaluate"],
        "保存模型": ["torch.save", "state_dict", "checkpoint"],
        "参数解析": ["argparse", "ArgumentParser"],
    }
    hits = {}
    for group, keys in patterns.items():
        group_hits = [f"L{i}: {l.strip()}" for i, l in enumerate(lines, 1) if any(k in l.strip() for k in keys)]
        if group_hits:
            hits[group] = group_hits
    return hits


def _summarize_python_file(file_path: str, content: str) -> str:
    source = _strip_line_numbers(content)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"### 文件分析: {file_path}\n- 无法解析 AST: {e}"

    module_doc = ast.get_docstring(tree)
    imports, classes, functions = [], [], []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(f"L{node.lineno}: import {', '.join(a.name for a in node.names)}")
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"L{node.lineno}: from {node.module or ''} import {', '.join(a.name for a in node.names)}")
        elif isinstance(node, ast.ClassDef):
            methods = [s.name for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append((node.name, node.lineno, methods))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node.name, node.lineno, [a.arg for a in node.args.args]))

    parts = [f"### 文件分析: {file_path}"]
    if module_doc:
        parts.append(f"- 模块说明: {module_doc.strip().splitlines()[0]}")
    if imports:
        parts.append("- 主要依赖:")
        for item in imports[:15]:
            parts.append(f"  - {item}")
    if classes:
        parts.append("- 类结构:")
        for name, ln, methods in classes:
            parts.append(f"  - L{ln}: class `{name}` -> 方法: {', '.join(methods[:10]) if methods else '无'}")
    if functions:
        parts.append("- 顶层函数:")
        for name, ln, args in functions[:20]:
            parts.append(f"  - L{ln}: `{name}({', '.join(args)})`")
    return "\n".join(parts)


def _summarize_text_file(file_path: str, content: str) -> str:
    source = _strip_line_numbers(content)
    lines = [line for line in source.splitlines() if line.strip()]
    preview = "\n".join(lines[:20])
    return f"### 文件分析: {file_path}\n- 这是一个文本文件。\n- 前 20 行预览:\n{preview}"


def _build_deterministic_report(delegation: str, latest_tool_results: list) -> str:
    if not latest_tool_results:
        return f"REPORT TO ARCHITECT:\n- **Task**: {delegation}\n- **Status**: Failed\n- **Issues Found**: No tool results."

    parts = ["REPORT TO ARCHITECT:", f"- **Task**: {delegation}", "- **Status**: Complete"]
    files_read, files_modified, issues, body = [], [], [], []

    for item in latest_tool_results:
        tool_name = item.get("tool_name", "unknown")
        tool_args = item.get("tool_args", {}) or {}
        content = item.get("content", "")

        if tool_name == "mcp_list_dir":
            body.append(_summarize_list_dir(content))
        elif tool_name == "mcp_read_file":
            fp = tool_args.get("file_path", "unknown")
            files_read.append(fp)
            body.append(_summarize_python_file(fp, content) if fp.endswith(".py") else _summarize_text_file(fp, content))
        elif tool_name == "mcp_write_file":
            fp = tool_args.get("file_path", "unknown")
            files_modified.append(fp)
            body.append(f"### 写文件结果\n- 已写入: {fp}\n- 工具返回: {content}")
        elif tool_name in ("memories_search", "memories_read", "memories_list", "add_ad_hoc_note"):
            body.append(f"### Memory 结果\n- Tool: {tool_name}\n- Args: {tool_args}\n- Output:\n{content[:3000]}")
        else:
            body.append(f"### 工具结果\n- Tool: {tool_name}\n- Args: {tool_args}\n- Content:\n{content[:4000]}")

        if isinstance(content, str) and content.startswith("Error:"):
            issues.append(content)

    parts.append(f"- **Files Read**: {', '.join(files_read) if files_read else 'None'}")
    parts.append(f"- **Files Modified**: {', '.join(files_modified) if files_modified else 'None'}")
    parts.append(f"- **Issues Found**: {'; '.join(issues) if issues else 'None'}")
    parts.append("")
    parts.append("\n\n".join(body))
    return "\n".join(parts)


def build_graph(
    architect_llm,
    coder_llm,
    mcp_tools: list,
    hook_engine: HookEngine = None,
    rag_tool=None,
    memory_tools: list = None,           # ★ 新增
    memory_injector=None,                 # ★ 新增
    max_tool_calls_per_turn: int = 15,
    workspace_dir: str = ".",
    file_tracker: FileTracker = None,
):
    if hook_engine is None:
        hook_engine = HookEngine()
    if file_tracker is None:
        file_tracker = FileTracker()
    if memory_tools is None:
        memory_tools = []

    # ★ Coder 拿到所有工具（mcp + rag + memory）
    all_tools = mcp_tools + ([rag_tool] if rag_tool else []) + memory_tools
    architect_bound = architect_llm
    coder_bound = coder_llm.bind_tools(all_tools)
    tool_map = {t.name: t for t in all_tools}

    turn_ctx = TurnContext(max_tool_calls=max_tool_calls_per_turn)

    async def architect_node(state: AgentState) -> dict:
        messages = await compress_history(state["messages"], architect_llm)
        nonlocal turn_ctx
        turn_ctx.phase = TurnPhase.ARCHITECTURE

        last_user = next(
            (m.content for m in reversed(messages)
             if isinstance(m, HumanMessage) and getattr(m, "name", "") != "SystemRuntime"),
            "",
        )
        skill = match_skill(last_user)

        project_ctx = f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        context_summary = file_tracker.build_context_summary()
        if context_summary:
            project_ctx += "\n" + context_summary

        # ★ 注入 memory_summary
        if memory_injector:
            mem_frag = memory_injector.build_system_prompt_fragment()
            if mem_frag:
                project_ctx += mem_frag

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
                content="[System]: You must either delegate to coder or answer the user and end with AWAITING USER INPUT.",
                name="SystemRuntime",
            )
            return {
                "messages": [res, guidance],
                "current_role": ROLE_ARCHITECT,
                "delegation": "",
                "tool_call_count": 0,
                "budget_exhausted": False,
                "latest_tool_results": [],
            }

        print(f"\n🏛️  Architect:\n{content}\n")

        delegation = ""
        if "DELEGATE TO CODER:" in content:
            raw = content.split("DELEGATE TO CODER:", 1)[-1].strip()
            delegation = raw.split("AWAITING USER INPUT")[0].strip()
            turn_ctx.phase = TurnPhase.DELEGATION

        if "AWAITING USER INPUT" in content:
            turn_ctx.complete(content)
            print(f"📊 {turn_ctx.to_event()}")

        return {
            "messages": [res],
            "current_role": ROLE_ARCHITECT,
            "delegation": delegation,
            "tool_call_count": 0,
            "budget_exhausted": False,
            "latest_tool_results": [],
        }

    async def coder_node(state: AgentState) -> dict:
        messages = await compress_history(state["messages"], architect_llm)
        delegation = state.get("delegation", "")
        nonlocal turn_ctx
        turn_ctx.phase = TurnPhase.EXECUTION

        sys_content = CODER_SYSTEM + f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        if delegation:
            sys_content += f"\n\n[CURRENT TASK FROM ARCHITECT]: {delegation}"

        context_summary = file_tracker.build_context_summary()
        if context_summary:
            sys_content += "\n" + context_summary

        sys_content += (
            "\n\n[PHASE]: TOOL EXECUTION PHASE.\n"
            "Your job is to call the required tool(s). Match the tool name in the task.\n"
            "If task says memories_search, call memories_search (NOT mcp_search_files).\n"
            "Call exactly one tool. Do not explain after deciding."
        )

        clean = [m for m in messages if not isinstance(m, SystemMessage)]
        msgs = [SystemMessage(content=sys_content)] + clean

        res = await coder_bound.ainvoke(msgs)
        content = (res.content or "").strip()

        if content:
            print(f"\n💻 Coder:\n{content}\n")
        else:
            print("\n🔧 Coder: thinking...")

        if res.tool_calls:
            print(f"🛠️  Tools: {[tc['name'] for tc in res.tool_calls]}")

        return {"messages": [res], "current_role": ROLE_CODER}

    async def hooked_tools_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not getattr(last_msg, "tool_calls", None):
            return {"messages": [], "latest_tool_results": []}

        async def execute_single(tool_call):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {}) or {}
            call_id = tool_call["id"]

            if tool_name == "mcp_write_file" and "file_path" not in tool_args:
                tool_args["file_path"] = "new_script.py"
                print(f"⚠️  [AutoFix] file_path auto-filled")

            # 仅对 mcp_ 工具做去重和已读检查
            if tool_name.startswith("mcp_") and file_tracker.is_duplicate_call(tool_name, tool_args):
                msg = f"[Duplicate call skipped] {tool_name}"
                return (ToolMessage(content=msg, tool_call_id=call_id),
                        {"tool_name": tool_name, "tool_args": tool_args, "content": msg})

            if tool_name == "mcp_read_file":
                fp = tool_args.get("file_path", "")
                snap = file_tracker._files.get(fp)
                if snap and not snap.is_stale:
                    summary = file_tracker.get_file_summary(fp)
                    msg = f"[Already read] {fp}. Summary: {summary}."
                    return (ToolMessage(content=msg, tool_call_id=call_id),
                            {"tool_name": tool_name, "tool_args": tool_args, "content": msg})

            hook_ctx = {"tool_name": tool_name, **tool_args}
            hook_result = hook_engine.evaluate(HookEvent.PRE_TOOL_USE, tool_name=tool_name, context=hook_ctx)
            if hook_result.should_block:
                msg = f"Blocked: {hook_result.block_reason}"
                return (ToolMessage(content=msg, tool_call_id=call_id),
                        {"tool_name": tool_name, "tool_args": tool_args, "content": msg})

            if tool_name not in tool_map:
                msg = f"Tool not found: {tool_name}"
                return (ToolMessage(content=msg, tool_call_id=call_id),
                        {"tool_name": tool_name, "tool_args": tool_args, "content": msg})

            try:
                tool_obj = tool_map[tool_name]
                result = await tool_obj.ainvoke(tool_args) if hasattr(tool_obj, "ainvoke") else tool_obj.invoke(tool_args)
                result_str = str(result)
            except Exception as e:
                result_str = f"Error: {e}"

            if tool_name.startswith("mcp_"):
                file_tracker.record_tool_call(tool_name, tool_args, result_preview=result_str[:200],
                                              success=not result_str.startswith("Error:"))
                if tool_name == "mcp_read_file" and not result_str.startswith("Error:"):
                    fp = tool_args.get("file_path", "unknown")
                    summary = _summarize_python_file(fp, result_str) if fp.endswith(".py") else f"{len(result_str.splitlines())} lines"
                    file_tracker.record_file_read(fp, result_str, summary[:500])
                elif tool_name == "mcp_list_dir" and not result_str.startswith("Error:"):
                    file_tracker.record_dir_listing(tool_args.get("directory", "."), result_str)
                elif tool_name in ("mcp_write_file", "mcp_apply_patch"):
                    fp = tool_args.get("file_path", "")
                    if fp: file_tracker.record_file_modified(fp)
                elif tool_name == "mcp_delete_file":
                    fp = tool_args.get("file_path", "")
                    if fp: file_tracker.record_file_deleted(fp)

            hook_engine.evaluate(HookEvent.POST_TOOL_USE, tool_name=tool_name, context={**hook_ctx, "result": result_str[:300]})

            return (ToolMessage(content=result_str, tool_call_id=call_id),
                    {"tool_name": tool_name, "tool_args": tool_args, "content": result_str})

        results = await asyncio.gather(*[execute_single(tc) for tc in last_msg.tool_calls])
        stats = file_tracker.get_stats()
        print(f"📊 [Context] files_read={stats['files_read']}, stale={stats['files_stale']}, tool_calls={stats['tool_calls_total']}")
        return {"messages": [x[0] for x in results], "latest_tool_results": [x[1] for x in results]}

    async def budget_node(state: AgentState) -> dict:
        nonlocal turn_ctx
        exhausted = turn_ctx.increment_tool_call()
        count = turn_ctx.tool_calls
        if exhausted:
            steering = HumanMessage(
                content="[System]: Budget exhausted. Output REPORT TO ARCHITECT.",
                name="SystemRuntime",
            )
            print(f"⚠️  [Budget] Exhausted at {count}/{turn_ctx.max_tool_calls}")
            return {"messages": [steering], "tool_call_count": count, "budget_exhausted": True}
        return {"tool_call_count": count}

    async def coder_report_node(state: AgentState) -> dict:
        delegation = state.get("delegation", "")
        latest_tool_results = state.get("latest_tool_results", [])
        report = _build_deterministic_report(delegation, latest_tool_results)
        context_summary = file_tracker.build_context_summary()
        if context_summary:
            report += "\n\n" + context_summary
        print(f"\n📋 Coder Report:\n{report[:4000]}\n")
        return {"messages": [AIMessage(content=report)], "current_role": ROLE_CODER_REPORT}

    def route_after_architect(state):
        last_msg = state["messages"][-1]
        delegation = state.get("delegation", "").strip()
        if getattr(last_msg, "name", "") == "SystemRuntime":
            return "architect"
        if delegation:
            return "coder"
        return END

    def route_after_coder(state):
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", "") or ""
        if getattr(last_msg, "tool_calls", None):
            return "hooked_tools"
        if "REPORT TO ARCHITECT" in content:
            return "architect"
        if content.strip():
            return "architect"
        return END

    def route_after_budget(state):
        return "coder_report"

    def route_after_coder_report(state):
        last_msg = state["messages"][-1]
        if getattr(last_msg, "name", "") == "SystemRuntime":
            return "coder_report"
        return "architect"

    builder = StateGraph(AgentState)
    builder.add_node("architect", architect_node)
    builder.add_node("coder", coder_node)
    builder.add_node("hooked_tools", hooked_tools_node)
    builder.add_node("budget", budget_node)
    builder.add_node("coder_report", coder_report_node)
    builder.set_entry_point("architect")
    builder.add_conditional_edges("architect", route_after_architect, {"coder": "coder", "architect": "architect", END: END})
    builder.add_conditional_edges("coder", route_after_coder, {"hooked_tools": "hooked_tools", "architect": "architect", END: END})
    builder.add_edge("hooked_tools", "budget")
    builder.add_conditional_edges("budget", route_after_budget, {"coder_report": "coder_report"})
    builder.add_conditional_edges("coder_report", route_after_coder_report, {"coder_report": "coder_report", "architect": "architect"})
    return builder.compile(checkpointer=MemorySaver())