"""
Tool Invoker - 统一执行闭环引擎。
流程：
schema校验 → 权限检查 → 已读复用 → turn内去重 → PreHook →
执行(超时/重试) → PostHook → 自愈 → 观测 → 审计
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from autocoder.harness.schema import ToolSchema, infer_schema_from_langchain_tool
from autocoder.harness.permissions import (
    PermissionPolicy, PermissionLevel, DEFAULT_TOOL_PERMISSIONS
)
from autocoder.harness.telemetry import TelemetryCollector, ToolSpan, SpanStatus
from autocoder.harness.self_heal import SelfHealAnalyzer, HealSuggestion
from autocoder.harness.audit import AuditLogger


@dataclass
class InvocationResult:
    tool_name: str
    args: Dict[str, Any]
    content: str
    success: bool
    span: ToolSpan
    heal: Optional[HealSuggestion] = None
    blocked: bool = False

    @property
    def error(self) -> Optional[str]:
        return self.span.error

    def to_tool_message_content(self) -> str:
        if self.heal and not self.success:
            return self.content + "\n\n" + self.heal.to_llm_prompt()
        return self.content


class ToolInvoker:
    def __init__(
        self,
        tool_map: Dict[str, Any],
        permission_policy: PermissionPolicy,
        hook_engine=None,
        file_tracker=None,
        audit_logger: Optional[AuditLogger] = None,
        default_timeout: float = 90.0,
        max_retries: int = 1,
        read_summarizer: Optional[Callable[[str, str], str]] = None,
    ):
        self.tool_map = tool_map
        self.policy = permission_policy
        self.hook_engine = hook_engine
        self.file_tracker = file_tracker
        self.audit = audit_logger
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.read_summarizer = read_summarizer
        self.healer = SelfHealAnalyzer()

        self._schemas: Dict[str, ToolSchema] = {}
        for name, obj in tool_map.items():
            try:
                self._schemas[name] = infer_schema_from_langchain_tool(obj)
            except Exception:
                self._schemas[name] = ToolSchema(name=name, description="", params=[])

    async def invoke(
        self,
        tool_name: str,
        args: Dict[str, Any],
        telemetry: TelemetryCollector,
    ) -> InvocationResult:
        perm = DEFAULT_TOOL_PERMISSIONS.get(tool_name, PermissionLevel.READ)
        span = telemetry.start_span(tool_name, args, permission_level=perm.name)

        if tool_name not in self.tool_map:
            return self._fail(span, args, f"Error: Tool not registered: {tool_name}", blocked=True)

        # Schema 校验
        schema = self._schemas.get(tool_name)
        if schema:
            ok, err, normalized = schema.validate(args)
            if not ok:
                return self._fail(span, args, f"Error: Schema validation failed: {err}", blocked=True)
            args = normalized

        span.metadata["normalized_args"] = args

        # 权限检查
        allowed, deny_reason = self.policy.check(tool_name, perm)
        if not allowed:
            return self._fail(span, args, f"Error: {deny_reason}", blocked=True)

        # 已读文件复用摘要
        if tool_name == "mcp_read_file" and self.file_tracker:
            fp = args.get("file_path", "")
            snap = self.file_tracker._files.get(fp)
            if snap and snap.was_read and not snap.is_stale:
                summary = self.file_tracker.get_file_summary(fp)
                msg = f"[Already read] {fp}. Summary: {summary}. Re-read only if the file changed."
                span.finish(SpanStatus.OK, result=msg)
                self._log(span)
                return InvocationResult(
                    tool_name=tool_name,
                    args=args,
                    content=msg,
                    success=True,
                    span=span,
                )

        # ★ 只做 turn 内去重，不再跨 turn 去重
        if self._is_duplicate_in_current_turn(telemetry, tool_name, args):
            msg = f"Error: [Duplicate call skipped in current turn] {tool_name}"
            span.finish(SpanStatus.BLOCKED, result=msg, error=msg)
            self._log(span)
            return InvocationResult(
                tool_name=tool_name,
                args=args,
                content=msg,
                success=False,
                span=span,
                blocked=True,
                heal=self.healer.analyze(tool_name, args, msg),
            )

        # PreToolUse Hook
        if self.hook_engine:
            from autocoder.orchestrator.hook_engine import HookEvent
            hook_ctx = {"tool_name": tool_name, **args}
            hook_res = self.hook_engine.evaluate(HookEvent.PRE_TOOL_USE, tool_name, hook_ctx)
            if hook_res.should_block:
                return self._fail(
                    span,
                    args,
                    f"Error: Blocked by hook: {hook_res.block_reason}",
                    blocked=True,
                )

        last_err = ""
        final_status = SpanStatus.ERROR

        for attempt in range(self.max_retries + 1):
            span.retry_count = attempt
            try:
                tool_obj = self.tool_map[tool_name]
                if hasattr(tool_obj, "ainvoke"):
                    coro = tool_obj.ainvoke(args)
                else:
                    coro = asyncio.to_thread(tool_obj.invoke, args)

                result = await asyncio.wait_for(coro, timeout=self.default_timeout)
                result_str = str(result)

                if result_str.startswith("Error"):
                    last_err = result_str
                    if not self._is_retryable(result_str) or attempt >= self.max_retries:
                        break
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue

                span.finish(
                    SpanStatus.OK if attempt == 0 else SpanStatus.RETRIED,
                    result=result_str,
                )
                self._post_hook(tool_name, args, result_str, success=True)
                self._log(span)
                return InvocationResult(
                    tool_name=tool_name,
                    args=args,
                    content=result_str,
                    success=True,
                    span=span,
                )

            except asyncio.CancelledError:
                span.finish(SpanStatus.ERROR, error="Cancelled")
                self._log(span)
                raise
            except asyncio.TimeoutError:
                last_err = f"Error: Timeout after {self.default_timeout}s"
                final_status = SpanStatus.TIMEOUT
                break
            except Exception as e:
                last_err = f"Error: {type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                break

        heal = self.healer.analyze(tool_name, args, last_err)
        span.finish(final_status, result=last_err, error=last_err)
        self._post_hook(tool_name, args, last_err, success=False)
        self._log(span)

        return InvocationResult(
            tool_name=tool_name,
            args=args,
            content=last_err,
            success=False,
            span=span,
            heal=heal,
        )

    async def invoke_many(
        self,
        calls: List[tuple],
        telemetry: TelemetryCollector,
    ) -> List[InvocationResult]:
        return await asyncio.gather(*[
            self.invoke(name, args, telemetry)
            for name, args in calls
        ])

    def _is_duplicate_in_current_turn(
        self,
        telemetry: TelemetryCollector,
        tool_name: str,
        args: Dict[str, Any],
    ) -> bool:
        """
        只检查当前 turn 的 telemetry spans。
        注意：当前 span 已经加入 telemetry.spans，所以要排除最后一个。
        """
        for s in telemetry.spans[:-1]:
            if s.tool_name != tool_name:
                continue
            prev_args = s.metadata.get("normalized_args")
            if prev_args == args:
                return True
        return False

    def _fail(
        self,
        span: ToolSpan,
        args: dict,
        msg: str,
        blocked: bool = False,
    ) -> InvocationResult:
        span.finish(
            SpanStatus.BLOCKED if blocked else SpanStatus.ERROR,
            result=msg,
            error=msg,
        )
        self._log(span)
        heal = self.healer.analyze(span.tool_name, args, msg)
        return InvocationResult(
            tool_name=span.tool_name,
            args=args,
            content=msg,
            success=False,
            span=span,
            heal=heal,
            blocked=blocked,
        )

    def _log(self, span: ToolSpan) -> None:
        if self.audit:
            self.audit.write(span)

    def _post_hook(
        self,
        tool_name: str,
        args: dict,
        result: str,
        success: bool,
    ) -> None:
        if self.hook_engine:
            from autocoder.orchestrator.hook_engine import HookEvent
            self.hook_engine.evaluate(
                HookEvent.POST_TOOL_USE,
                tool_name,
                {
                    "tool_name": tool_name,
                    **args,
                    "result": result[:300],
                    "success": success,
                },
            )

        if not self.file_tracker or not tool_name.startswith("mcp_"):
            return

        self.file_tracker.record_tool_call(
            tool_name,
            args,
            result_preview=result[:200],
            success=success,
        )

        if not success:
            return

        fp = args.get("file_path", "")

        if tool_name == "mcp_read_file" and fp:
            if self.read_summarizer:
                summary = self.read_summarizer(fp, result)
            else:
                summary = f"{len(result.splitlines())} lines"
            self.file_tracker.record_file_read(fp, result, summary[:500])

        elif tool_name == "mcp_list_dir":
            self.file_tracker.record_dir_listing(args.get("directory", "."), result)

        elif tool_name == "mcp_write_file" and fp:
            self.file_tracker.record_file_modified(fp)

        elif tool_name == "mcp_append_file" and fp:
            self.file_tracker.record_file_appended(fp)

        elif tool_name == "mcp_apply_patch" and fp:
            self.file_tracker.record_file_modified(fp)

        elif tool_name == "mcp_delete_file" and fp:
            self.file_tracker.record_file_deleted(fp)

    @staticmethod
    def _is_retryable(error: str) -> bool:
        low = error.lower()
        non_retry = [
            "not found",
            "permission",
            "blocked",
            "invalid",
            "duplicate",
            "missing",
            "must be",
            "read-only",
            "appears",
            "access denied",
        ]
        return not any(k in low for k in non_retry)