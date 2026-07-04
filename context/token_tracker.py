"""
Token 追踪器 - 工业级实现，对齐 Codex context_manager/history.rs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple

# ── tiktoken 可选依赖 ─────────────────────────────────
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    _ENCODER = None
    _USE_TIKTOKEN = False


# ═════════════════════════════════════════════════════
# Layer 1: 底层 token 计数
# ═════════════════════════════════════════════════════

def approx_token_count(text: str) -> int:
    """
    对齐 Codex codex_utils_output_truncation::approx_token_count
    优先 tiktoken (cl100k_base), 失败回退到 UTF-8 bytes // 4
    """
    if not text:
        return 0
    if _USE_TIKTOKEN:
        try:
            return len(_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    # 回退：Codex 就是 bytes.len() / 4，四舍五入向上
    byte_len = len(text.encode("utf-8", errors="ignore"))
    return max(1, (byte_len + 3) // 4)


def approx_bytes_for_tokens(tokens: int) -> int:
    """对齐 Codex approx_bytes_for_tokens: tokens * 4"""
    return max(0, tokens) * 4


def truncate_text_by_tokens(text: str, max_tokens: int) -> Tuple[str, bool]:
    """
    对齐 Codex truncate_text with TruncationPolicy::Tokens
    返回 (截断后文本, 是否发生截断)
    """
    if max_tokens <= 0 or not text:
        return "", bool(text)

    current = approx_token_count(text)
    if current <= max_tokens:
        return text, False

    # 目标字节数 (预留 20% 给尾部与省略号)
    total_bytes = approx_bytes_for_tokens(max_tokens)
    head_bytes = int(total_bytes * 0.75)
    tail_bytes = int(total_bytes * 0.15)

    text_bytes = text.encode("utf-8", errors="ignore")
    if len(text_bytes) <= head_bytes + tail_bytes:
        return text, False

    head = text_bytes[:head_bytes].decode("utf-8", errors="ignore")
    tail = text_bytes[-tail_bytes:].decode("utf-8", errors="ignore")
    omitted = current - approx_token_count(head) - approx_token_count(tail)
    marker = f"\n\n...[Truncated ~{omitted} tokens by output policy]...\n\n"
    return head + marker + tail, True


# ═════════════════════════════════════════════════════
# Layer 2: 单条 message 估算
# ═════════════════════════════════════════════════════

MESSAGE_OVERHEAD_TOKENS = 4


def estimate_message_tokens(msg: Any) -> int:
    """
    对齐 Codex estimate_item_token_count
    覆盖 content + tool_calls + overhead
    """
    total = MESSAGE_OVERHEAD_TOKENS

    # 1. content
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        total += approx_token_count(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                total += approx_token_count(str(text))
            else:
                total += approx_token_count(str(part))
    else:
        total += approx_token_count(str(content))

    # 2. tool_calls (AIMessage)
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args = tc.get("args", {})
            else:
                name = getattr(tc, "name", "")
                args = getattr(tc, "args", {})
            total += approx_token_count(str(name)) + approx_token_count(str(args))

    # 3. tool_call_id (ToolMessage)
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        total += approx_token_count(str(tool_call_id))

    return total


def sum_messages_tokens(messages: list) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


# ═════════════════════════════════════════════════════
# Layer 3: 顶层 TokenTracker + Breakdown
# ═════════════════════════════════════════════════════

@dataclass
class TokenBreakdown:
    """按 message 角色分类的 token 统计"""
    system: int = 0
    human: int = 0
    ai: int = 0
    tool: int = 0
    tool_calls_meta: int = 0
    total_estimated: int = 0

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "human": self.human,
            "ai": self.ai,
            "tool": self.tool,
            "tool_calls_meta": self.tool_calls_meta,
            "total_estimated": self.total_estimated,
        }

    @classmethod
    def from_messages(cls, messages: list) -> "TokenBreakdown":
        bd = cls()
        for m in messages:
            cls_name = type(m).__name__
            tokens = estimate_message_tokens(m)
            bd.total_estimated += tokens
            if cls_name == "SystemMessage":
                bd.system += tokens
            elif cls_name == "HumanMessage":
                bd.human += tokens
            elif cls_name == "AIMessage":
                bd.ai += tokens
                if getattr(m, "tool_calls", None):
                    bd.tool_calls_meta += tokens
            elif cls_name == "ToolMessage":
                bd.tool += tokens
        return bd


@dataclass
class TokenTracker:
    """
    完整 Token 追踪引擎。
    对齐 Codex ContextManager (token 部分) + TokenUsageInfo。
    """
    # ── 上次 API 真实返回 ──
    last_api_input_tokens: Optional[int] = None
    last_api_output_tokens: Optional[int] = None
    last_api_total_tokens: Optional[int] = None

    # ── 累积统计 ──
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    total_api_calls: int = 0

    # ── 配置 ──
    model_context_window: int = 8192
    auto_compact_ratio: float = 0.75
    hard_limit_ratio: float = 0.90
    max_tool_output_tokens: int = 2000  # 单个工具输出上限

    # ── 内部：最近一次估算，避免重复计算 ──
    _last_breakdown: Optional[TokenBreakdown] = field(default=None, init=False, repr=False)

    def record_ai_message(self, ai_message: Any) -> None:
        """
        从 LangChain AIMessage 提取 usage。
        """
        usage = None

        # Path 1: usage_metadata (标准)
        um = getattr(ai_message, "usage_metadata", None)
        if um and isinstance(um, dict) and um.get("total_tokens"):
            usage = {
                "input_tokens": um.get("input_tokens", 0),
                "output_tokens": um.get("output_tokens", 0),
                "total_tokens": um.get("total_tokens", 0),
            }

        # Path 2: response_metadata.token_usage (Ollama 常见)
        if usage is None:
            rm = getattr(ai_message, "response_metadata", None)
            if rm and isinstance(rm, dict):
                tu = rm.get("token_usage") or {}
                if tu.get("total_tokens") or tu.get("prompt_tokens"):
                    usage = {
                        "input_tokens": tu.get("prompt_tokens", 0),
                        "output_tokens": tu.get("completion_tokens", 0),
                        "total_tokens": tu.get("total_tokens", 0)
                            or (tu.get("prompt_tokens", 0) + tu.get("completion_tokens", 0)),
                    }

        if usage is None:
            return

        self.last_api_input_tokens = usage["input_tokens"]
        self.last_api_output_tokens = usage["output_tokens"]
        self.last_api_total_tokens = usage["total_tokens"]
        self.cumulative_input_tokens += usage["input_tokens"]
        self.cumulative_output_tokens += usage["output_tokens"]
        self.total_api_calls += 1

    def estimate_total(self, messages: list) -> int:
        """
        对齐 Codex: 有 API 记录: last_api_total + 增量估算; 无 API 记录: 全部估算
        """
        breakdown = TokenBreakdown.from_messages(messages)
        self._last_breakdown = breakdown

        if self.last_api_total_tokens is None:
            return breakdown.total_estimated

        last_ai_idx = -1
        for i, m in enumerate(messages):
            if type(m).__name__ == "AIMessage":
                last_ai_idx = i

        if last_ai_idx < 0:
            return breakdown.total_estimated

        new_messages = messages[last_ai_idx + 1:]
        new_tokens = sum_messages_tokens(new_messages)
        return self.last_api_total_tokens + new_tokens

    def should_compact(self, messages: list) -> bool:
        total = self.estimate_total(messages)
        return total >= int(self.model_context_window * self.auto_compact_ratio)

    def is_hard_limit_reached(self, messages: list) -> bool:
        total = self.estimate_total(messages)
        return total >= int(self.model_context_window * self.hard_limit_ratio)

    def get_status(self, messages: list) -> dict:
        total = self.estimate_total(messages)
        limit = self.model_context_window
        bd = self._last_breakdown or TokenBreakdown()
        return {
            "total": total,
            "limit": limit,
            "usage_pct": round(total / limit * 100, 1) if limit else 0,
            "api_last_total": self.last_api_total_tokens,
            "api_last_input": self.last_api_input_tokens,
            "api_last_output": self.last_api_output_tokens,
            "api_calls": self.total_api_calls,
            "cum_input": self.cumulative_input_tokens,
            "cum_output": self.cumulative_output_tokens,
            "breakdown": bd.to_dict(),
        }

    def format_status_line(self, messages: list) -> str:
        s = self.get_status(messages)
        bd = s["breakdown"]
        api_str = (
            f"api={s['api_last_total']}({s['api_last_input']}↑/{s['api_last_output']}↓)"
            if s["api_last_total"] is not None else "api=none"
        )
        return (
            f"🔎 [Token] {s['total']:,}/{s['limit']:,} ({s['usage_pct']}%) "
            f"| {api_str} calls={s['api_calls']} "
            f"| sys={bd['system']} human={bd['human']} ai={bd['ai']} tool={bd['tool']}"
        )

    def format_audit_panel(self) -> str:
        return (
            "\n╔══════ Token Audit ══════╗\n"
            f"║ API Calls:    {self.total_api_calls}\n"
            f"║ Cum Input:    {self.cumulative_input_tokens:,}\n"
            f"║ Cum Output:   {self.cumulative_output_tokens:,}\n"
            f"║ Cum Total:    {self.cumulative_input_tokens + self.cumulative_output_tokens:,}\n"
            f"║ Last Total:   {self.last_api_total_tokens}\n"
            "╚═════════════════════════╝"
        )