"""
工具结果格式化器 (借鉴 Codex `FunctionCallError::RespondToModel` + Claude Code `emit_metrics`)

将所有工具输出和错误统一格式化为 LLM 可理解的结构化消息。
"""

class ToolResultFormatter:
    """
    格式化工具执行结果。
    借鉴 Codex：错误变成自然语言消息 (RespondToModel)
    借鉴 Claude Code：结构化指标输出
    """

    def format_success(self, tool_name: str, arguments: dict, result: str) -> str:
        """格式化成功的工具调用"""
        return f"Success: {result}"

    def format_error(self, tool_name: str, arguments: dict, error: str) -> str:
        """
        格式化工具错误。
        借鉴 Codex `FunctionCallError::RespondToModel`：
        错误消息会被 LLM 直接看到，帮助它理解失败原因并调整策略。
        """
        return f"Error invoking tool '{tool_name}' with arguments {arguments}: {error}. Please fix the error and try again."

    def format_report(self, call_history: list[dict]) -> str:
        """从调用历史生成结构化汇报"""
        if not call_history:
            return "REPORT TO ARCHITECT: Task complete. (no tools called)"

        parts = []
        for call in call_history:
            status = "✓" if call["success"] else "✗"
            tool = call.get("tool", "unknown")
            args = call.get("args", {})
            output = call.get("output", "")[:100]
            parts.append(f"[{status}] {tool}({args}) → {output}")

        return "REPORT TO ARCHITECT: Task complete.\n" + "\n".join(parts)