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

load_dotenv()

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from autocoder.utils.config import Config
from autocoder.agent.state_machine import build_graph
from autocoder.rag.retriever import rag_search
from autocoder.orchestrator.hook_engine import (
    HookEngine, HookEvent, Rule, Condition, HookAction
)
from autocoder.context.file_tracker import FileTracker

from autocoder.memory.models import MemoriesConfig
from autocoder.memory.store import MemoryStore
from autocoder.memory.extractor import StageOneExtractor
from autocoder.memory.consolidator import PhaseTwoConsolidator
from autocoder.memory.tools import create_memory_tools
from autocoder.memory.rollout_recorder import RolloutRecorder
from autocoder.memory.startup import MemoryStartupPipeline

file_tracker = FileTracker()


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
        res = await session.call_tool(
            "search_files",
            arguments={"regex": regex, "file_pattern": file_pattern}
        )
        return res.content[0].text

    @tool
    async def mcp_execute_bash(command: str) -> str:
        """Execute a shell command in the workspace."""
        res = await session.call_tool("execute_bash", arguments={"command": command})
        return res.content[0].text

    @tool
    async def mcp_write_file(file_path: str, content: str) -> str:
        """Write content to a file, creating parent directories if needed."""
        res = await session.call_tool(
            "write_file",
            arguments={"file_path": file_path, "content": content}
        )
        return res.content[0].text

    @tool
    async def mcp_apply_patch(file_path: str, original: str, replacement: str) -> str:
        """Apply a targeted patch: replace exact text in a file."""
        res = await session.call_tool(
            "apply_patch",
            arguments={
                "file_path": file_path,
                "original": original,
                "replacement": replacement
            }
        )
        return res.content[0].text

    @tool
    async def mcp_delete_file(file_path: str) -> str:
        """Delete a file in the workspace."""
        res = await session.call_tool("delete_file", arguments={"file_path": file_path})
        return res.content[0].text

    return [
        mcp_list_dir,
        mcp_read_file,
        mcp_search_files,
        mcp_execute_bash,
        mcp_write_file,
        mcp_apply_patch,
        mcp_delete_file,
    ]


async def read_input(prompt: str = "🧑 You: ") -> str | None:
    def _read():
        try:
            return input(prompt)
        except EOFError:
            return None
    return await asyncio.to_thread(_read)


async def main():
    print("🚀 AutoCoder v6 Starting (Codex-style automatic memory)...")
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

        # Codex-style memory config
        memories_config = MemoriesConfig(
            generate_memories=True,
            use_memories=True,
            enable_phase2=True,
            use_vector_search=False,        # Windows + chroma 1.5.x 先禁掉
            auto_startup_pipeline=True,
            min_rollout_idle_hours=0,       # 方便你本地立即重启测试
            max_rollouts_per_startup=8,
            max_rollout_age_days=30,
            max_unused_days=30,
        )
        memory_store = MemoryStore(workspace, memories_config)
        memory_tools = create_memory_tools(memory_store, workspace)

        stage1_extractor = StageOneExtractor(
            llm=coder_llm,
            config=memories_config,
            store=memory_store,
        )
        phase2_consolidator = PhaseTwoConsolidator(
            llm=architect_llm,
            config=memories_config,
            store=memory_store,
        )

        rollout_recorder = RolloutRecorder(workspace)
        startup_pipeline = MemoryStartupPipeline(
            workspace_dir=workspace,
            config=memories_config,
            store=memory_store,
            extractor=stage1_extractor,
            consolidator=phase2_consolidator,
        )

        # 启动时自动扫描和提取历史 rollout
        await startup_pipeline.run_once(active_session_id=rollout_recorder.session_id)

        graph = build_graph(
            architect_llm=architect_llm,
            coder_llm=coder_llm,
            mcp_tools=mcp_tools,
            hook_engine=hook_engine,
            rag_tool=rag_search,
            memory_store=memory_store,
            memory_tools=memory_tools,
            max_tool_calls_per_turn=15,
            workspace_dir=str(workspace),
            file_tracker=file_tracker,
        )

        from autocoder.tasks import (
            TaskScheduler, SessionTaskContext, RegularTask,
            EventBus, TurnAbortReason
        )
        from autocoder.models.turn import TurnContext, TurnStatus

        lang_config = {"configurable": {"thread_id": "main_session"}}
        event_bus = EventBus()
        session_ctx = SessionTaskContext(
            graph=graph,
            lang_config=lang_config,
            event_bus=event_bus,
        )
        scheduler = TaskScheduler(session_ctx)

        print(f"🧠 Memory: {memory_store.get_memory_count()} memories loaded")
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
                        print("\n⚠️ Ctrl+C detected, interrupting...")
                        await scheduler.abort_all_tasks(TurnAbortReason.INTERRUPTED)

                # 只记录 rollout；不在这里做长期记忆提取
                if ctx.status == TurnStatus.COMPLETED:
                    new_tool_records = []
                    for r in file_tracker._tool_history[before_tool_count:]:
                        new_tool_records.append({
                            "tool_name": r.tool_name,
                            "result_preview": r.result_preview,
                            "success": r.success,
                            "timestamp": r.timestamp,
                        })

                    rollout_recorder.append_turn(
                        turn_id=ctx.turn_id,
                        user_input=user_input,
                        assistant_response=ctx.last_agent_message or "",
                        tool_records=new_tool_records,
                        cwd=str(workspace),
                        file_stats=file_tracker.get_stats(),
                    )

            except KeyboardInterrupt:
                print("\n⚠️ Interrupting current turn...")
                await scheduler.abort_all_tasks(TurnAbortReason.INTERRUPTED)
            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())