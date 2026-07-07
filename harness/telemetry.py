"""
可观测性 - OpenTelemetry 风格 Span/Trace。
每次工具调用产生一个 Span，包含完整输入输出元数据。
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"
    RETRIED = "retried"
    TIMEOUT = "timeout"


@dataclass
class ToolSpan:
    trace_id: str
    span_id: str
    tool_name: str
    args: Dict[str, Any]
    status: SpanStatus = SpanStatus.OK
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_ms: Optional[float] = None
    result_preview: str = ""
    error: Optional[str] = None
    retry_count: int = 0
    permission_level: str = "READ"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, status: SpanStatus, result: str = "", error: str = "") -> None:
        # 幂等保护：只允许 finish 一次
        if self.ended_at is not None:
            return
        self.ended_at = time.time()
        self.duration_ms = round((self.ended_at - self.started_at) * 1000, 2)
        self.status = status
        self.result_preview = (result or "")[:200]
        if error:
            self.error = error[:500]

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryCollector:
    """Turn 级别的 Span 收集器（每个 turn 独立实例，无全局污染）"""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.spans: List[ToolSpan] = []

    def start_span(self, tool_name: str, args: Dict[str, Any],
                   permission_level: str = "READ") -> ToolSpan:
        span = ToolSpan(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:8],
            tool_name=tool_name,
            args=self._safe_args(args),
            permission_level=permission_level,
        )
        self.spans.append(span)
        return span

    @staticmethod
    def _safe_args(args: Dict[str, Any]) -> Dict[str, Any]:
        """裁剪超长参数值，防止审计日志爆炸"""
        safe = {}
        for k, v in (args or {}).items():
            if isinstance(v, str) and len(v) > 200:
                safe[k] = v[:200] + f"...[+{len(v) - 200} chars]"
            else:
                safe[k] = v
        return safe

    def summarize(self) -> dict:
        total = len(self.spans)
        by_status: Dict[str, int] = {}
        by_tool: Dict[str, int] = {}
        total_ms = 0.0
        for s in self.spans:
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
            by_tool[s.tool_name] = by_tool.get(s.tool_name, 0) + 1
            total_ms += s.duration_ms or 0.0
        return {
            "trace_id": self.trace_id,
            "total_calls": total,
            "total_duration_ms": round(total_ms, 2),
            "by_status": by_status,
            "by_tool": by_tool,
            "errors": [s.error for s in self.spans if s.error],
        }

    def format_table(self) -> str:
        """人类可读的 Span 表格"""
        if not self.spans:
            return "(no tool spans)"
        icons = {"ok": "✓", "error": "✗", "blocked": "🚫", "retried": "↻", "timeout": "⏱"}
        lines = [f"╔══ Trace {self.trace_id} " + "═" * 30]
        for s in self.spans:
            icon = icons.get(s.status.value, "?")
            dur = s.duration_ms or 0.0  # ★ 修复：未 finish 的 span 不能崩溃
            lines.append(
                f"║ {icon} {s.tool_name:<25} {dur:>7.1f}ms  "
                f"[{s.permission_level}] retry={s.retry_count}"
            )
            if s.error:
                lines.append(f"║   └─ {s.error[:80]}")
        lines.append("╚" + "═" * 52)
        return "\n".join(lines)