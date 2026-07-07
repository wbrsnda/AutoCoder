import asyncio
import os
import platform
import ast
import re
from typing import Annotated, TypedDict

from langchain_core.messages import (
    HumanMessage, SystemMessage, ToolMessage, AIMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from autocoder.agent.prompts import ARCHITECT_SYSTEM, CODER_SYSTEM
from autocoder.memory.compress import compress_history_if_needed
from autocoder.context.token_tracker import (
    TokenTracker, truncate_text_by_tokens, estimate_message_tokens
)
from autocoder.orchestrator.hook_engine import HookEngine
from autocoder.skills.builtin import match_skill
from autocoder.context.file_tracker import FileTracker
from autocoder.models.turn import TurnContext, TurnPhase
from autocoder.utils.config import Config
from autocoder.harness import (
    ToolInvoker, PermissionPolicy, TelemetryCollector,
    AuditLogger, ContextGateway, PlannerGuard,
)

OS_NAME = platform.system()
ROLE_ARCHITECT = "architect"
ROLE_CODER = "coder"
ROLE_CODER_REPORT = "coder_report"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int
    current_role: str
    delegation: str
    budget_exhausted: bool
    latest_tool_results: list
    guard_retries: int
    turn_id: str


# ══════════════════════════════════════════════════════
# 辅助函数（保留原实现）
# ══════════════════════════════════════════════════════
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
        success = item.get("success", not str(content).startswith("Error"))

        # ★ 失败结果优先按失败处理，避免把 Duplicate/Error 当成空目录解析
        if not success or str(content).startswith("Error"):
            issues.append(str(content)[:300])
            body.append(
                f"### 工具调用失败\n"
                f"- Tool: {tool_name}\n"
                f"- Args: {tool_args}\n"
                f"- Error:\n{str(content)[:1000]}"
            )
            continue

        if tool_name == "mcp_list_dir":
            body.append(_summarize_list_dir(content))
        elif tool_name == "mcp_read_file":
            fp = tool_args.get("file_path", "unknown")
            files_read.append(fp)
            body.append(_summarize_python_file(fp, content) if fp.endswith(".py") else _summarize_text_file(fp, content))
        elif tool_name in ("mcp_write_file", "mcp_append_file"):
            fp = tool_args.get("file_path", "unknown")
            files_modified.append(fp)
            verb = "写入" if tool_name == "mcp_write_file" else "追加"
            body.append(f"### {verb}文件结果\n- 目标: {fp}\n- 工具返回: {content}")
        elif tool_name == "mcp_apply_patch":
            fp = tool_args.get("file_path", "unknown")
            files_modified.append(fp)
            body.append(f"### Patch 结果\n- 目标: {fp}\n- 工具返回: {content}")
        elif tool_name in ("memories_search", "memories_read", "memories_list", "add_ad_hoc_note"):
            body.append(f"### Memory 结果\n- Tool: {tool_name}\n- Args: {tool_args}\n- Output:\n{content[:3000]}")
        else:
            body.append(f"### 工具结果\n- Tool: {tool_name}\n- Args: {tool_args}\n- Content:\n{content[:4000]}")

        if not success:
            issues.append(str(content)[:300])

    parts.append(f"- **Files Read**: {', '.join(files_read) if files_read else 'None'}")
    parts.append(f"- **Files Modified**: {', '.join(files_modified) if files_modified else 'None'}")
    parts.append(f"- **Issues Found**: {'; '.join(issues) if issues else 'None'}")
    parts.append("")
    parts.append("\n\n".join(body))
    return "\n".join(parts)


# ══════════════════════════════════════════════════════
# build_graph
# ══════════════════════════════════════════════════════
def build_graph(
    architect_llm,
    coder_llm,
    mcp_tools: list,
    hook_engine: HookEngine = None,
    rag_tool=None,
    memory_tools: list = None,
    memory_injector=None,
    max_tool_calls_per_turn: int = 15,
    workspace_dir: str = ".",
    file_tracker: FileTracker = None,
    token_tracker: TokenTracker = None,
    audit_logger: AuditLogger = None,
    permission_policy: PermissionPolicy = None,
):
    if hook_engine is None:
        hook_engine = HookEngine()
    if file_tracker is None:
        file_tracker = FileTracker()
    if memory_tools is None:
        memory_tools = []
    if token_tracker is None:
        token_tracker = TokenTracker(
            model_context_window=Config.MODEL_CONTEXT_WINDOW,
            auto_compact_ratio=Config.AUTO_COMPACT_TOKEN_RATIO,
            hard_limit_ratio=Config.HARD_LIMIT_RATIO,
            max_tool_output_tokens=Config.MAX_TOOL_OUTPUT_TOKENS,
        )
    if permission_policy is None:
        permission_policy = PermissionPolicy.from_sandbox_mode(
            os.getenv("AUTOCODER_SANDBOX", "workspace_write")
        )

    all_tools = mcp_tools + ([rag_tool] if rag_tool else []) + memory_tools
    architect_bound = architect_llm
    tool_map = {t.name: t for t in all_tools}

    # ── Harness 装配 ──
    def _read_summarizer(fp: str, content: str) -> str:
        if fp.endswith(".py"):
            return _summarize_python_file(fp, content)
        return f"{len(content.splitlines())} lines"

    invoker = ToolInvoker(
        tool_map=tool_map,
        permission_policy=permission_policy,
        hook_engine=hook_engine,
        file_tracker=file_tracker,
        audit_logger=audit_logger,
        default_timeout=90.0,
        max_retries=1,
        read_summarizer=_read_summarizer,
    )
    gateway = ContextGateway(all_tools)
    planner_guard = PlannerGuard(
    tool_names=[t.name for t in all_tools],
    file_tracker=file_tracker,
    )

    # per-turn telemetry（带上限清理，防内存泄漏）
    turn_telemetry: dict = {}

    def _get_telemetry(turn_id: str) -> TelemetryCollector:
        if turn_id not in turn_telemetry:
            if len(turn_telemetry) > 8:
                oldest = next(iter(turn_telemetry))
                turn_telemetry.pop(oldest, None)
            turn_telemetry[turn_id] = TelemetryCollector(trace_id=turn_id or None)
        return turn_telemetry[turn_id]

    # ── Turn 上下文 ──
    turn_state = {"ctx": TurnContext(max_tool_calls=max_tool_calls_per_turn)}

    def _ensure_fresh_turn_ctx(state_turn_id: str) -> None:
        current_ctx = turn_state["ctx"]
        if state_turn_id and state_turn_id != current_ctx.turn_id:
            print(f"🔄 [TurnCtx] Reset {current_ctx.turn_id} → {state_turn_id}")
            new_ctx = TurnContext(max_tool_calls=max_tool_calls_per_turn)
            new_ctx.turn_id = state_turn_id
            turn_state["ctx"] = new_ctx

    def _get_ctx() -> TurnContext:
        return turn_state["ctx"]

    def _record_usage(res):
        before_calls = token_tracker.total_api_calls
        token_tracker.record_ai_message(res)
        if token_tracker.total_api_calls > before_calls:
            _get_ctx().token_usage.add_api_call(
                token_tracker.last_api_input_tokens or 0,
                token_tracker.last_api_output_tokens or 0,
                token_tracker.last_api_total_tokens or 0,
            )

    # ══════════════════════════════════════════════════
    # ARCHITECT NODE
    # ══════════════════════════════════════════════════
    async def architect_node(state: AgentState) -> dict:
        raw_msgs = state["messages"]
        _ensure_fresh_turn_ctx(state.get("turn_id", ""))
        turn_ctx = _get_ctx()

        print(f"\n{'─'*60}")
        print(f"🔎 [ArchIN] messages={len(raw_msgs)} turn={turn_ctx.turn_id} tool_calls={turn_ctx.tool_calls}")
        print(token_tracker.format_status_line(raw_msgs))

        remove_updates, prompt_msgs, was_compressed = await compress_history_if_needed(
            raw_msgs, architect_llm, token_tracker, Config.KEEP_RECENT_MESSAGES
        )
        if was_compressed:
            turn_ctx.token_usage.compactions += 1

        turn_ctx.phase = TurnPhase.ARCHITECTURE

        last_user = next(
            (m.content for m in reversed(prompt_msgs)
             if isinstance(m, HumanMessage) and getattr(m, "name", "") != "SystemRuntime"),
            "",
        )

        # 判断当前是否是 Coder Report 回流到 Architect
        last_msg = raw_msgs[-1] if raw_msgs else None
        last_content = getattr(last_msg, "content", "") or ""
        has_coder_report = isinstance(last_msg, AIMessage) and str(last_content).lstrip().startswith("REPORT TO ARCHITECT")

        # ★ PlannerGuard：高置信度工具需求，直接强制 delegate，避免 Architect 假装“正在查”
        forced_delegation = planner_guard.pre_delegate(
            user_text=last_user,
            has_coder_report=has_coder_report,
        )
        if forced_delegation:
            content = f"DELEGATE TO CODER: {forced_delegation}"
            msg = AIMessage(content=content)
            turn_ctx.phase = TurnPhase.DELEGATION

            print(f"\n🛡️  [PlannerGuard] Forced delegation:\n{content}\n")

            return {
                "messages": remove_updates + [msg],
                "current_role": ROLE_ARCHITECT,
                "delegation": forced_delegation,
                "tool_call_count": 0,
                "budget_exhausted": False,
                "latest_tool_results": [],
                "guard_retries": 0,
            }

        # Coder report 回来后，不再注入 skill，避免再次触发同一技能导致循环
        skill = None if has_coder_report else match_skill(last_user)

        project_ctx = f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        context_summary = file_tracker.build_context_summary()
        if context_summary:
            project_ctx += "\n" + context_summary
        if memory_injector:
            mem_frag = memory_injector.build_system_prompt_fragment()
            if mem_frag:
                project_ctx += mem_frag

        sys_content = ARCHITECT_SYSTEM.replace("__PROJECT_CONTEXT__", project_ctx)
        if skill:
            sys_content += "\n\n[SKILL ACTIVATED]:\n" + "\n".join(skill["steps"])

        clean = [m for m in prompt_msgs if not isinstance(m, SystemMessage)]
        msgs = [SystemMessage(content=sys_content)] + clean

        res = await architect_bound.ainvoke(msgs)
        _record_usage(res)

        content = (res.content or "").strip()

        # ★ PlannerGuard：修正 Architect 违规输出
        normalized = planner_guard.normalize_architect_output(
            user_text=last_user,
            content=content,
            has_coder_report=has_coder_report,
        )
        if normalized != content:
            print(f"🛡️  [PlannerGuard] Architect output normalized:\nFROM:\n{content}\nTO:\n{normalized}\n")
            content = normalized
            res = AIMessage(content=content)

        guard_retries = state.get("guard_retries", 0)
        if not content or content == "None":
            if guard_retries >= Config.MAX_GUARD_RETRIES:
                fail_msg = AIMessage(content="模型生成回复中断。请您继续输入或指示下一步。")
                return {
                    "messages": remove_updates + [fail_msg],
                    "current_role": ROLE_ARCHITECT,
                    "delegation": "",
                    "tool_call_count": 0,
                    "budget_exhausted": False,
                    "latest_tool_results": [],
                    "guard_retries": 0,
                }
            guidance = HumanMessage(
                content="[System]: You must either delegate to coder or answer the user and end with AWAITING USER INPUT.",
                name="SystemRuntime",
            )
            return {
                "messages": remove_updates + [res, guidance],
                "current_role": ROLE_ARCHITECT,
                "delegation": "",
                "tool_call_count": 0,
                "budget_exhausted": False,
                "latest_tool_results": [],
                "guard_retries": guard_retries + 1,
            }

        print(f"\n🏛️  Architect:\n{content}\n")

        delegation = ""
        if "DELEGATE TO CODER:" in content:
            raw = content.split("DELEGATE TO CODER:", 1)[-1].strip()
            delegation = raw.split("AWAITING USER INPUT")[0].strip()
            # 不再强行只取第一行：
            # mcp_write_file / mcp_append_file / mcp_apply_patch 可能携带多行内容。
            turn_ctx.phase = TurnPhase.DELEGATION

        if "AWAITING USER INPUT" in content and not delegation:
            turn_ctx.complete(content)
            print(f"📊 {turn_ctx.to_event()}")
            print(token_tracker.format_audit_panel())
            tel = turn_telemetry.get(turn_ctx.turn_id)
            if tel and tel.spans:
                summary = tel.summarize()
                print(f"🔭 [Harness] calls={summary['total_calls']} "
                      f"total={summary['total_duration_ms']}ms "
                      f"status={summary['by_status']}")

        return {
            "messages": remove_updates + [res],
            "current_role": ROLE_ARCHITECT,
            "delegation": delegation,
            "tool_call_count": 0,
            "budget_exhausted": False,
            "latest_tool_results": [],
            "guard_retries": 0,
        }

    # ══════════════════════════════════════════════════
    # CODER NODE（★ Gateway 按需暴露工具）
    # ══════════════════════════════════════════════════
    async def coder_node(state: AgentState) -> dict:
        raw_msgs = state["messages"]
        turn_ctx = _get_ctx()

        print(token_tracker.format_status_line(raw_msgs))

        remove_updates, prompt_msgs, was_compressed = await compress_history_if_needed(
            raw_msgs, architect_llm, token_tracker, Config.KEEP_RECENT_MESSAGES
        )
        if was_compressed:
            turn_ctx.token_usage.compactions += 1

        delegation = state.get("delegation", "")
        turn_ctx.phase = TurnPhase.EXECUTION

        # ★ Deterministic ToolCall Fast Path
        # 对简单明确的 delegation，直接生成 tool_call，不再依赖 Coder 模型的 function calling 稳定性。
        deterministic_call = planner_guard.parse_delegation_to_tool_call(delegation)
        if deterministic_call and deterministic_call["name"] in tool_map:
            ai = AIMessage(content="", tool_calls=[deterministic_call])
            print(f"\n🧭 [PlannerGuard] Deterministic tool_call:\n{deterministic_call}\n")
            return {
                "messages": remove_updates + [ai],
                "current_role": ROLE_CODER,
            }

        # ★ Gateway 按需选择工具
        visible_tools = gateway.select_for_delegation(delegation)
        coder_bound_dyn = coder_llm.bind_tools(visible_tools)
        visible_manifest = gateway.build_visible_manifest(visible_tools)

        if len(visible_tools) < len(all_tools):
            print(f"🚪 [Gateway] Exposing {len(visible_tools)}/{len(all_tools)} tools")

        sys_content = CODER_SYSTEM + f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        if delegation:
            sys_content += f"\n\n[CURRENT TASK FROM ARCHITECT]: {delegation}"

        sys_content += "\n\n" + visible_manifest

        context_summary = file_tracker.build_context_summary()
        if context_summary:
            sys_content += "\n" + context_summary

        sys_content += (
            "\n\n[PHASE]: TOOL EXECUTION PHASE.\n"
            "Your job is to call the required tool(s). Match the tool name in the task.\n"
            "If task says memories_search, call memories_search (NOT mcp_search_files).\n"
            "Prefer mcp_append_file over mcp_write_file when the user asks to 'append' or extend a file.\n"
            "Call exactly one tool. Do not explain after deciding."
        )

        clean = [m for m in prompt_msgs if not isinstance(m, SystemMessage)]
        msgs = [SystemMessage(content=sys_content)] + clean

        res = await coder_bound_dyn.ainvoke(msgs)
        _record_usage(res)

        content = (res.content or "").strip()
        if content:
            print(f"\n💻 Coder:\n{content[:400]}{'...' if len(content) > 400 else ''}\n")
        else:
            print("\n🔧 Coder: thinking...")

        # ★ Coder Fallback：LLM 未产出 tool_calls 时，兜底再尝试确定性解析
        if not res.tool_calls and delegation:
            fallback_call = planner_guard.parse_delegation_to_tool_call(delegation)
            if fallback_call and fallback_call["name"] in tool_map:
                print(f"🧭 [PlannerGuard] Coder LLM failed, using deterministic fallback → {fallback_call['name']}")
                ai = AIMessage(content="", tool_calls=[fallback_call])
                return {
                    "messages": remove_updates + [ai],
                    "current_role": ROLE_CODER,
                }

        if res.tool_calls:
            print(f"🛠️  Tools: {[tc['name'] for tc in res.tool_calls]}")
        else:
            print("⚠️  [CoderGuard] Coder returned no tool_calls AND no fallback matched. Routing back to Architect.")

        return {
            "messages": remove_updates + [res],
            "current_role": ROLE_CODER,
        }

    # ══════════════════════════════════════════════════
    # HOOKED TOOLS NODE（★ 全部走 Invoker 执行闭环）
    # ══════════════════════════════════════════════════
    async def hooked_tools_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not getattr(last_msg, "tool_calls", None):
            return {"messages": [], "latest_tool_results": []}

        turn_id = state.get("turn_id", "") or _get_ctx().turn_id
        telemetry = _get_telemetry(turn_id)

        # 组装调用（保留原有的 file_path 兜底行为）
        calls = []
        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = dict(tc.get("args", {}) or {})
            if tool_name in ("mcp_write_file", "mcp_append_file") and "file_path" not in tool_args:
                tool_args["file_path"] = "new_script.py"
            calls.append((tool_name, tool_args))

        results = await invoker.invoke_many(calls, telemetry)

        tool_msgs = []
        tool_results = []
        for tc, result in zip(last_msg.tool_calls, results):
            content = result.to_tool_message_content()

            # ★ 保留原有 token 截断（自愈建议一并计入长度）
            probe = ToolMessage(content=content, tool_call_id=tc["id"])
            if estimate_message_tokens(probe) > token_tracker.max_tool_output_tokens:
                truncated, was_cut = truncate_text_by_tokens(
                    content, token_tracker.max_tool_output_tokens
                )
                if was_cut:
                    print(f"✂️  [ToolTruncate] {result.tool_name}")
                    content = truncated

            tool_msgs.append(ToolMessage(content=content, tool_call_id=tc["id"]))
            tool_results.append({
                "tool_name": result.tool_name,
                "tool_args": result.args,
                "content": result.content,
                "success": result.success,
                "duration_ms": result.span.duration_ms,
            })

        # 可观测性输出
        print("\n" + telemetry.format_table())
        stats = file_tracker.get_stats()
        print(
            f"📊 [Context] files_read={stats['files_read']}, "
            f"stale={stats['files_stale']}, "
            f"changed_unread={stats['changed_unread']}, "
            f"tool_calls={stats['tool_calls_total']}"
        )

        return {
            "messages": tool_msgs,
            "latest_tool_results": tool_results,
        }

    # ══════════════════════════════════════════════════
    # BUDGET / REPORT NODES（保留原实现）
    # ══════════════════════════════════════════════════
    async def budget_node(state: AgentState) -> dict:
        turn_ctx = _get_ctx()
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

    # ══════════════════════════════════════════════════
    # ROUTING（保留原实现）
    # ══════════════════════════════════════════════════
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