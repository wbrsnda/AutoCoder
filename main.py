import os
os.environ["ALLOWED_MSGPACK_MODULES"] = (
    "autocoder.models.turn:TurnStatus,"
    "autocoder.models.turn:TurnPhase"
)

import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from contextlib import AsyncExitStack

# 显式加载 autocoder/ 目录下的 .env
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from autocoder.utils.config import Config
from autocoder.agent.state_machine import build_graph
from autocoder.rag.retriever import rag_search
from autocoder.orchestrator.hook_engine import HookEngine, HookEvent, Rule, Condition, HookAction
from autocoder.context.file_tracker import FileTracker

from autocoder.memory import (
    MemoryWorkspace, MemoryRecorder, MemoryConsolidator,
    MemoryInjector, create_memory_tools,
)

from autocoder.context.token_tracker import TokenTracker
from autocoder.harness import AuditLogger, PermissionPolicy

file_tracker = FileTracker()

token_tracker = TokenTracker(
    model_context_window=Config.MODEL_CONTEXT_WINDOW,
    auto_compact_ratio=Config.AUTO_COMPACT_TOKEN_RATIO,
    hard_limit_ratio=Config.HARD_LIMIT_RATIO,
    max_tool_output_tokens=Config.MAX_TOOL_OUTPUT_TOKENS,
)
print(f"🔢 Token: window={Config.MODEL_CONTEXT_WINDOW} "
      f"compact@{Config.AUTO_COMPACT_TOKEN_RATIO*100:.0f}% "
      f"hard@{Config.HARD_LIMIT_RATIO*100:.0f}% "
      f"tool_cap={Config.MAX_TOOL_OUTPUT_TOKENS}")

CONSOLIDATE_EVERY_N_TURNS = 3


async def create_mcp_tools(session):
    @tool
    async def mcp_list_dir(directory: str = ".") -> str:
        """List directory contents in the workspace."""
        res = await session.call_tool("list_dir", arguments={"directory": directory})
        return res.content[0].text

    @tool
    async def mcp_read_file(file_path: str, start_line: int = 1, end_line: int = None) -> str:
        """Read file content with optional line range."""
        args = {"file_path": file_path, "start_line": start_line}
        if end_line is not None:
            args["end_line"] = end_line
        res = await session.call_tool("read_file", arguments=args)
        return res.content[0].text

    @tool
    async def mcp_search_files(regex: str, file_pattern: str = "*.*") -> str:
        """Search for a regex pattern in project files."""
        res = await session.call_tool("search_files", arguments={"regex": regex, "file_pattern": file_pattern})
        return res.content[0].text

    @tool
    async def mcp_execute_bash(command: str) -> str:
        """Execute a shell command in the workspace."""
        res = await session.call_tool("execute_bash", arguments={"command": command})
        return res.content[0].text

    @tool
    async def mcp_write_file(file_path: str, content: str) -> str:
        """
        OVERWRITE-write a file. This REPLACES existing file content.
        For appending content without losing existing data, use mcp_append_file instead.
        """
        res = await session.call_tool("write_file", arguments={"file_path": file_path, "content": content})
        return res.content[0].text

    @tool
    async def mcp_append_file(file_path: str, content: str) -> str:
        """
        APPEND content to a file. Creates the file if it does not exist.
        Use this when you want to add new content while KEEPING existing content.
        """
        res = await session.call_tool(
            "append_file",
            arguments={"file_path": file_path, "content": content, "add_newline": True},
        )
        return res.content[0].text

    @tool
    async def mcp_apply_patch(file_path: str, original: str, replacement: str) -> str:
        """Apply a targeted patch: replace exact text in a file."""
        res = await session.call_tool(
            "apply_patch",
            arguments={"file_path": file_path, "original": original, "replacement": replacement},
        )
        return res.content[0].text

    @tool
    async def mcp_delete_file(file_path: str) -> str:
        """Delete a file in the workspace."""
        res = await session.call_tool("delete_file", arguments={"file_path": file_path})
        return res.content[0].text

    @tool
    async def mcp_write_files(files: list[dict]) -> str:
        """Batch create/overwrite multiple files. files=[{"file_path":"a.py","content":"..."}]"""
        res = await session.call_tool("write_files", arguments={"files": files})
        return res.content[0].text

    # ── Batch 1: Project development tools ──

    @tool
    async def mcp_find_files(pattern: str = "*.*", directory: str = ".", max_depth: int = None) -> str:
        """Find files by glob pattern. Use this to discover project structure."""
        args = {"pattern": pattern, "directory": directory}
        if max_depth is not None:
            args["max_depth"] = max_depth
        res = await session.call_tool("find_files", arguments=args)
        return res.content[0].text

    @tool
    async def mcp_create_directory(path: str) -> str:
        """Create a directory (and parent dirs if needed)."""
        res = await session.call_tool("create_directory", arguments={"path": path})
        return res.content[0].text

    @tool
    async def mcp_move_file(source: str, destination: str) -> str:
        """Move or rename a file/directory within the workspace."""
        res = await session.call_tool("move_file", arguments={"source": source, "destination": destination})
        return res.content[0].text

    @tool
    async def mcp_move_files(sources: list[str], destination_dir: str) -> str:
        """Batch move multiple files into a target directory."""
        res = await session.call_tool("move_files", arguments={"sources": sources, "destination_dir": destination_dir})
        return res.content[0].text

    @tool
    async def mcp_git_status() -> str:
        """Show the working tree status (staged, unstaged, untracked changes)."""
        res = await session.call_tool("git_status", arguments={})
        return res.content[0].text

    @tool
    async def mcp_git_diff(file_path: str = "", staged: bool = False) -> str:
        """Show git diff for a file or all changes."""
        res = await session.call_tool("git_diff", arguments={"file_path": file_path, "staged": staged})
        return res.content[0].text

    return [
        mcp_list_dir, mcp_read_file, mcp_search_files, mcp_execute_bash,
        mcp_write_file, mcp_append_file, mcp_apply_patch, mcp_delete_file, mcp_write_files,
        mcp_find_files, mcp_create_directory, mcp_move_file, mcp_move_files,
        mcp_git_status, mcp_git_diff,
    ]


async def read_input(prompt: str = "🧑 You: ") -> str | None:
    def _read():
        try:
            return input(prompt)
        except EOFError:
            return None
    return await asyncio.to_thread(_read)


async def main():
    print("🚀 AutoCoder v7 Starting (Tool Harness enabled)...")
    Config.apply_proxy()

    workspace = Config.WORKSPACE_DIR
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["AUTOCODER_WORKSPACE"] = str(workspace)

    server_path = Path(__file__).parent / "mcp_server" / "mcp_server.py"
    if not server_path.exists():
        raise FileNotFoundError(f"MCP Server not found: {server_path}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=os.environ.copy()
    )

    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
        await session.initialize()
        mcp_tools = await create_mcp_tools(session)

        hook_engine = HookEngine()
        hooks_json = Path(__file__).parent / "plugins" / "hooks.json"
        hook_engine.load_from_json(hooks_json)
        hook_engine.register(Rule(
            name="fallback-block-rm-rf",
            event=HookEvent.PRE_TOOL_USE,
            action=HookAction.BLOCK,
            matcher="mcp_execute_bash",
            conditions=[Condition(field="command", operator="regex_match", pattern=r"rm\s+-rf")],
            message="[Fallback] Dangerous 'rm -rf' blocked.",
        ))

        # ★★★ Harness 装配 ★★★
        audit_logger = AuditLogger(log_dir=workspace / ".autocoder" / "audit")
        permission_policy = PermissionPolicy.from_sandbox_mode(
            os.getenv("AUTOCODER_SANDBOX", "workspace_write")
        )
        print(f"🛡️  Sandbox: {permission_policy.max_level.name} "
              f"| Audit: {workspace / '.autocoder' / 'audit'}")

        architect_llm = ChatOpenAI(
            model=Config.ARCHITECT_MODEL,
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            temperature=0.1,
        )
        coder_llm = ChatOpenAI(
            model=Config.CODER_MODEL,
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            temperature=0.0,
        )

        mem_workspace = MemoryWorkspace(workspace)
        mem_workspace.ensure_initialized()
        mem_recorder = MemoryRecorder(mem_workspace)
        mem_consolidator = MemoryConsolidator(mem_workspace, architect_llm)
        mem_injector = MemoryInjector(mem_workspace)
        mem_tools = create_memory_tools(workspace)

        summary_preview = mem_workspace.read_file("memory_summary.md").strip()
        if summary_preview:
            print(f"🧠 Memory: loaded summary ({len(summary_preview)} chars)")
        else:
            print("🧠 Memory: no prior summary (fresh start)")

        graph = build_graph(
            architect_llm=architect_llm,
            coder_llm=coder_llm,
            mcp_tools=mcp_tools,
            hook_engine=hook_engine,
            rag_tool=rag_search,
            memory_tools=mem_tools,
            memory_injector=mem_injector,
            max_tool_calls_per_turn=15,
            workspace_dir=str(workspace),
            file_tracker=file_tracker,
            token_tracker=token_tracker,
            audit_logger=audit_logger,            # ★
            permission_policy=permission_policy,  # ★
        )

        from autocoder.tasks import (
            TaskScheduler, SessionTaskContext, RegularTask, EventBus, TurnAbortReason
        )
        from autocoder.models.turn import TurnContext, TurnStatus

        lang_config = {"configurable": {"thread_id": "main_session"}}
        event_bus = EventBus()
        session_ctx = SessionTaskContext(graph=graph, lang_config=lang_config, event_bus=event_bus)
        scheduler = TaskScheduler(session_ctx)

        print(f"✅ Ready. Workspace: {workspace}")
        print("💡 Enter your request. Ctrl+C to interrupt. 'exit' to quit.\n")

        while True:
            try:
                user_input = await read_input()
                if user_input is None:
                    print("\n👋 EOF detected. Goodbye!")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    await scheduler.abort_all_tasks(TurnAbortReason.REPLACED)
                    print("👋 Goodbye!")
                    break

                if scheduler.is_active:
                    scheduler.queue_pending_input(user_input)
                    print("📨 Turn 进行中，已加入待处理队列")
                    continue

                before_tool_count = len(file_tracker._tool_history)

                ctx = TurnContext(max_tool_calls=15)
                await scheduler.spawn_task(RegularTask(), ctx, user_input)

                if scheduler._active is not None:
                    try:
                        await scheduler._active.done.wait()
                    except KeyboardInterrupt:
                        print("\n⚠️ Ctrl+C interrupting...")
                        await scheduler.abort_all_tasks(TurnAbortReason.INTERRUPTED)

                if ctx.status == TurnStatus.COMPLETED:
                    new_tools = [
                        {"tool_name": r.tool_name, "tool_args": {}, "success": r.success}
                        for r in file_tracker._tool_history[before_tool_count:]
                    ]
                    mem_recorder.record_turn(
                        user_input=user_input,
                        architect_response=ctx.last_agent_message or "",
                        tool_calls=new_tools,
                    )

                    if mem_recorder.turn_count > 0 and mem_recorder.turn_count % CONSOLIDATE_EVERY_N_TURNS == 0:
                        print(f"\n🧠 [Memory] Triggering consolidation (turn {mem_recorder.turn_count})...")
                        try:
                            await mem_consolidator.consolidate()
                        except Exception as e:
                            print(f"⚠️  [Memory] Consolidation error: {e}")

            except KeyboardInterrupt:
                print("\n⚠️ Interrupting...")
                await scheduler.abort_all_tasks(TurnAbortReason.INTERRUPTED)
            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())