import asyncio
import operator
import platform
import ast
import re
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from autocoder.agent.prompts import ARCHITECT_SYSTEM, CODER_SYSTEM
from autocoder.memory.compress import compress_history
from autocoder.orchestrator.hook_engine import HookEngine, HookEvent
from autocoder.skills.builtin import match_skill

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
    """把 mcp_read_file 返回的 `1 | code` 格式还原成源码。"""
    cleaned = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+\s*\|\s?(.*)$", line)
        cleaned.append(match.group(1) if match else line)
    return "\n".join(cleaned)


def _summarize_list_dir(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    files = []
    dirs = []
    for line in lines:
        if line.startswith("[FILE] "):
            files.append(line.replace("[FILE] ", "", 1))
        elif line.startswith("[DIR] "):
            dirs.append(line.replace("[DIR] ", "", 1))

    parts = []
    parts.append("### 目录内容")
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
        "数据加载": ["Dataset", "DataLoader", "read_csv", "torch.load", "np.load", "__getitem__", "__len__"],
        "模型定义": ["nn.Module", "TransformerEncoder", "Linear(", "LayerNorm", "Softplus", "ReLU"],
        "训练循环": ["model.train()", "optimizer.zero_grad()", "loss.backward()", "optimizer.step()", "for epoch"],
        "验证评估": ["model.eval()", "torch.no_grad()", "evaluate", "mae", "obo", "valid", "test"],
        "保存模型": ["torch.save", "state_dict", "checkpoint", "os.replace"],
        "参数解析": ["argparse", "ArgumentParser", "add_argument"],
    }

    hits = {}
    for group, keys in patterns.items():
        group_hits = []
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if any(k in stripped for k in keys):
                group_hits.append(f"L{idx}: {stripped}")
        if group_hits:
            hits[group] = group_hits
    return hits


def _summarize_python_file(file_path: str, content: str) -> str:
    source = _strip_line_numbers(content)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return (
            f"### 文件分析: {file_path}\n"
            f"- 无法解析 AST: {e}\n"
            f"- 这通常表示读取内容不完整或源码本身存在语法问题。"
        )

    module_doc = ast.get_docstring(tree)

    imports = []
    classes = []
    functions = []
    has_main = False

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
            imports.append(f"L{node.lineno}: import {', '.join(names)}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            imports.append(f"L{node.lineno}: from {mod} import {', '.join(names)}")
        elif isinstance(node, ast.ClassDef):
            methods = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(sub.name)
            classes.append((node.name, node.lineno, methods))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            functions.append((node.name, node.lineno, args))

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                for comp in test.comparators:
                    if isinstance(comp, ast.Constant) and comp.value == "__main__":
                        has_main = True

    patterns = _collect_code_patterns(source)

    parts = []
    parts.append(f"### 文件分析: {file_path}")

    if module_doc:
        first_doc = module_doc.strip().splitlines()[0]
        parts.append(f"- 模块说明: {first_doc}")

    if imports:
        parts.append("- 主要依赖:")
        for item in imports[:15]:
            parts.append(f"  - {item}")

    if classes:
        parts.append("- 类结构:")
        for name, lineno, methods in classes:
            method_text = ", ".join(methods[:15]) if methods else "无明显方法"
            parts.append(f"  - L{lineno}: class `{name}` -> 方法: {method_text}")

    if functions:
        parts.append("- 顶层函数:")
        for name, lineno, args in functions[:20]:
            arg_text = ", ".join(args)
            parts.append(f"  - L{lineno}: `{name}({arg_text})`")

    if patterns:
        parts.append("- 代码模式证据:")
        for group, hits in patterns.items():
            parts.append(f"  - {group}:")
            for h in hits[:6]:
                parts.append(f"    - {h}")

    judgement = []

    function_names = {name for name, _, _ in functions}
    class_names = {name for name, _, _ in classes}

    if "train" in function_names:
        judgement.append("该文件包含 `train()` 入口，属于训练主流程或训练入口文件。")
    if "test_model" in function_names or "evaluate_ivac_style" in function_names:
        judgement.append("该文件包含测试/评估逻辑，不只是训练，还负责模型评估。")
    if "StrongDensityTransformer" in class_names:
        judgement.append("文件中定义了 `StrongDensityTransformer` 模型，是核心模型结构。")
    if "FullFeatDataset" in class_names:
        judgement.append("文件中定义了数据集类 `FullFeatDataset`，负责从特征文件和标注 CSV 中加载数据。")
    if has_main:
        judgement.append("文件包含 `if __name__ == '__main__'`，可直接作为脚本运行。")
    if "保存模型" in patterns:
        judgement.append("文件中包含 checkpoint 保存逻辑。")

    if judgement:
        parts.append("- 综合判断:")
        for j in judgement:
            parts.append(f"  - {j}")

    return "\n".join(parts)


def _summarize_text_file(file_path: str, content: str) -> str:
    source = _strip_line_numbers(content)
    lines = [line for line in source.splitlines() if line.strip()]
    preview = "\n".join(lines[:20])
    return (
        f"### 文件分析: {file_path}\n"
        f"- 这是一个文本文件。\n"
        f"- 前 20 行预览:\n{preview}"
    )


def _build_deterministic_report(delegation: str, latest_tool_results: list) -> str:
    if not latest_tool_results:
        return (
            "REPORT TO ARCHITECT:\n"
            f"- **Task**: {delegation}\n"
            "- **Status**: Failed\n"
            "- **Issues Found**: No tool results were captured for this task."
        )

    parts = []
    parts.append("REPORT TO ARCHITECT:")
    parts.append(f"- **Task**: {delegation}")
    parts.append("- **Status**: Complete")

    files_read = []
    files_modified = []
    issues = []

    body = []

    for item in latest_tool_results:
        tool_name = item.get("tool_name", "unknown")
        tool_args = item.get("tool_args", {}) or {}
        content = item.get("content", "")

        if tool_name == "mcp_list_dir":
            body.append(_summarize_list_dir(content))

        elif tool_name == "mcp_read_file":
            file_path = tool_args.get("file_path", "unknown")
            files_read.append(file_path)

            if file_path.endswith(".py"):
                body.append(_summarize_python_file(file_path, content))
            else:
                body.append(_summarize_text_file(file_path, content))

        elif tool_name == "mcp_write_file":
            file_path = tool_args.get("file_path", "unknown")
            files_modified.append(file_path)
            body.append(f"### 写文件结果\n- 已写入: {file_path}\n- 工具返回: {content}")

        elif tool_name == "mcp_apply_patch":
            file_path = tool_args.get("file_path", "unknown")
            files_modified.append(file_path)
            body.append(f"### Patch 结果\n- 已修改: {file_path}\n- 工具返回: {content}")

        elif tool_name == "mcp_delete_file":
            file_path = tool_args.get("file_path", "unknown")
            files_modified.append(file_path)
            body.append(f"### 删除结果\n- 已删除: {file_path}\n- 工具返回: {content}")

        else:
            body.append(
                f"### 工具结果\n"
                f"- Tool: {tool_name}\n"
                f"- Args: {tool_args}\n"
                f"- Content:\n{content[:4000]}"
            )

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
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, HumanMessage)
                and getattr(m, "name", "") != "SystemRuntime"
            ),
            "",
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

        return {
            "messages": [res],
            "current_role": ROLE_ARCHITECT,
            "delegation": delegation,
            "tool_call_count": 0,
            "budget_exhausted": False,
            "latest_tool_results": [],
        }

    # ── Coder Act 节点：只负责调用工具 ─────────────────────────
    async def coder_node(state: AgentState) -> dict:
        messages = await compress_history(state["messages"], architect_llm)
        delegation = state.get("delegation", "")

        sys_content = CODER_SYSTEM + f"\n\n[WORKSPACE]: {workspace_dir}\n[OS]: {OS_NAME}"
        if delegation:
            sys_content += f"\n\n[CURRENT TASK FROM ARCHITECT]: {delegation}"

        sys_content += (
            "\n\n[PHASE]: TOOL EXECUTION PHASE.\n"
            "Your job in this phase is ONLY to call the required tool(s).\n"
            "If the task needs file content, call mcp_read_file once.\n"
            "If the task needs a directory listing, call mcp_list_dir once.\n"
            "If the task needs file creation, call mcp_write_file once.\n"
            "Do not explain after deciding to call tools."
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
            current_names = [tc["name"] for tc in res.tool_calls]
            print(f"🛠️  Tools: {current_names}")

        return {
            "messages": [res],
            "current_role": ROLE_CODER,
        }

    # ── hooked_tools_node ─────────────────────────────────────
    async def hooked_tools_node(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        if not getattr(last_msg, "tool_calls", None):
            return {"messages": [], "latest_tool_results": []}

        async def execute_single(tool_call):
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            call_id = tool_call["id"]

            hook_ctx = {"tool_name": tool_name, **tool_args}
            hook_result = hook_engine.evaluate(
                HookEvent.PRE_TOOL_USE,
                tool_name=tool_name,
                context=hook_ctx,
            )

            if hook_result.should_block:
                blocked_msg = f"Blocked: {hook_result.block_reason}"
                return (
                    ToolMessage(content=blocked_msg, tool_call_id=call_id),
                    {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "content": blocked_msg,
                    },
                )

            if tool_name not in tool_map:
                not_found = f"Tool not found: {tool_name}"
                return (
                    ToolMessage(content=not_found, tool_call_id=call_id),
                    {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "content": not_found,
                    },
                )

            try:
                tool_obj = tool_map[tool_name]
                result = (
                    await tool_obj.ainvoke(tool_args)
                    if hasattr(tool_obj, "ainvoke")
                    else tool_obj.invoke(tool_args)
                )
                result_str = str(result)
            except Exception as e:
                result_str = f"Error: {e}"

            hook_engine.evaluate(
                HookEvent.POST_TOOL_USE,
                tool_name=tool_name,
                context={**hook_ctx, "result": result_str[:300]},
            )

            return (
                ToolMessage(content=result_str, tool_call_id=call_id),
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "content": result_str,
                },
            )

        results = await asyncio.gather(*[execute_single(tc) for tc in last_msg.tool_calls])

        tool_messages = [x[0] for x in results]
        structured_results = [x[1] for x in results]

        return {
            "messages": tool_messages,
            "latest_tool_results": structured_results,
        }

    # ── Budget 节点 ───────────────────────────────────────────
    async def budget_node(state: AgentState) -> dict:
        count = state.get("tool_call_count", 0) + 1
        if count >= max_tool_calls_per_turn:
            steering = HumanMessage(
                content="[System]: Budget exhausted. Output REPORT TO ARCHITECT.",
                name="SystemRuntime",
            )
            return {
                "messages": [steering],
                "tool_call_count": count,
                "budget_exhausted": True,
            }
        return {"tool_call_count": count}

    # ── Coder Report 节点：确定性报告 ──────────────────────────
    async def coder_report_node(state: AgentState) -> dict:
        delegation = state.get("delegation", "")
        latest_tool_results = state.get("latest_tool_results", [])

        report = _build_deterministic_report(delegation, latest_tool_results)

        print(f"\n📋 Coder Report (deterministic):\n{report[:4000]}\n")

        return {
            "messages": [AIMessage(content=report)],
            "current_role": ROLE_CODER_REPORT,
        }

    # ── 路由函数 ──────────────────────────────────────────────
    def route_after_architect(state: AgentState):
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", "") or ""
        delegation = state.get("delegation", "").strip()

        if getattr(last_msg, "name", "") == "SystemRuntime":
            return "architect"

        if delegation:
            return "coder"

        if "AWAITING USER INPUT" in content:
            return END

        return END

    def route_after_coder(state: AgentState):
        last_msg = state["messages"][-1]
        content = getattr(last_msg, "content", "") or ""

        if getattr(last_msg, "tool_calls", None):
            return "hooked_tools"

        if "REPORT TO ARCHITECT" in content:
            return "architect"

        if content.strip():
            return "architect"

        return END

    def route_after_budget(state: AgentState):
        return "coder_report"

    def route_after_coder_report(state: AgentState):
        last_msg = state["messages"][-1]

        if getattr(last_msg, "name", "") == "SystemRuntime":
            return "coder_report"

        return "architect"

    # ── 构建图 ────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    builder.add_node("architect", architect_node)
    builder.add_node("coder", coder_node)
    builder.add_node("hooked_tools", hooked_tools_node)
    builder.add_node("budget", budget_node)
    builder.add_node("coder_report", coder_report_node)

    builder.set_entry_point("architect")

    builder.add_conditional_edges(
        "architect",
        route_after_architect,
        {
            "coder": "coder",
            "architect": "architect",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "coder",
        route_after_coder,
        {
            "hooked_tools": "hooked_tools",
            "architect": "architect",
            END: END,
        },
    )

    builder.add_edge("hooked_tools", "budget")

    builder.add_conditional_edges(
        "budget",
        route_after_budget,
        {
            "coder_report": "coder_report",
        },
    )

    builder.add_conditional_edges(
        "coder_report",
        route_after_coder_report,
        {
            "coder_report": "coder_report",
            "architect": "architect",
        },
    )

    return builder.compile(checkpointer=MemorySaver())