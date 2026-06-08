# autocoder/state/turn_state.py
"""
Turn 阶段枚举。

借鉴 Codex TurnLifecycleState：
- 阶段是显式枚举，不依赖 LLM 输出的魔法词
- TurnState 类已删除，其职责由 LangGraph AgentState 承担
- TurnPhase 枚举保留供路由函数引用
"""
from enum import Enum


class TurnPhase(str, Enum):
    """
    Turn 的执行阶段。
    借鉴 Codex agent_turn_running / TurnLifecycleState。
    """
    ARCHITECT  = "architect"    # 架构师规划中
    CODER      = "coder"        # Coder 执行中
    CONFIRMING = "confirming"   # 等待用户确认（如删除确认）
    COMPLETED  = "completed"    # 任务完成
    IDLE       = "idle"         # 等待用户输入