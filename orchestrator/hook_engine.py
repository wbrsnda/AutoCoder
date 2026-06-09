# autocoder/orchestrator/hook_engine.py
"""
Hook 引擎。

借鉴来源：
- Claude Code rule_engine.py：blocking_rules vs warning_rules 分离，
  hookSpecificOutput.permissionDecision = "deny" 是真正的阻断格式
- Claude Code config_loader.py：Condition dataclass，operator 枚举，
  从 JSON/Markdown 加载规则
- Claude Code hooks.json：声明式配置，action: block|warn|log
- Codex hook_runtime.rs：PreToolUse → execute → PostToolUse 三段式，
  should_block + block_reason 的返回格式

与原实现的差异：
- 删除 hooks/engine.py（重复）
- HookConfig 升级为 Rule dataclass，支持 Condition 列表
- 新增 load_from_json()，真正加载 plugins/hooks.json
- evaluate() 返回结构化结果，而非 Optional[str]
"""
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ── 枚举 ──────────────────────────────────────────────────────

class HookEvent(str, Enum):
    PRE_TOOL_USE  = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP          = "Stop"


class HookAction(str, Enum):
    BLOCK = "block"   # 阻断工具调用，返回错误给 LLM
    WARN  = "warn"    # 注入警告 SystemMessage，不阻断
    LOG   = "log"     # 仅打印日志，不影响执行


# ── 数据类 ────────────────────────────────────────────────────

@dataclass
class Condition:
    """
    单个匹配条件。
    借鉴 Claude Code config_loader.py Condition dataclass。
    """
    field:    str   # "command" | "file_path" | "content" | ...
    operator: str   # "regex_match" | "contains" | "equals" | "not_contains"
    pattern:  str   # 匹配模式


@dataclass
class Rule:
    """
    单条 Hook 规则。
    借鉴 Claude Code config_loader.py Rule dataclass。
    """
    name:         str
    event:        HookEvent
    action:       HookAction          = HookAction.LOG
    matcher:      Optional[str]       = None   # 工具名匹配，None = 全部
    conditions:   list[Condition]     = field(default_factory=list)
    message:      str                 = ""
    enabled:      bool                = True


@dataclass
class HookResult:
    """
    Hook 评估结果。
    借鉴 Codex hook_runtime.rs PreToolUseOutcome：
    should_block + block_reason + additional_contexts
    """
    should_block:  bool          = False
    block_reason:  Optional[str] = None
    warn_message:  Optional[str] = None


# ── 正则缓存（借鉴 rule_engine.py 的 lru_cache）─────────────

@lru_cache(maxsize=128)
def _compile_regex(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# ── Hook 引擎 ─────────────────────────────────────────────────

class HookEngine:
    """
    Hook 引擎。

    借鉴 Claude Code hookify 插件的完整设计：
    - 声明式规则（从 JSON 加载）
    - Condition 列表（ALL 条件匹配才触发）
    - blocking_rules vs warning_rules 分离
    - 硬编码兜底规则（防止 hooks.json 被删除后失去保护）
    """

    def __init__(self):
        self._rules: list[Rule] = []

    # ── 规则注册 ──────────────────────────────────────────────

    def register(self, rule: Rule) -> None:
        """注册单条规则（用于硬编码兜底）"""
        self._rules.append(rule)

    def load_from_json(self, json_path: str | Path) -> None:
        """
        从 hooks.json 加载声明式规则。
        借鉴 Claude Code config_loader.py load_rules()。

        hooks.json 格式：
        {
          "hooks": {
            "PreToolUse": [
              {
                "name": "block-rm-rf",
                "matcher": "mcp_execute_bash",
                "action": "block",
                "message": "Dangerous command blocked.",
                "conditions": [
                  {"field": "command", "operator": "regex_match", "pattern": "rm\\s+-rf"}
                ]
              }
            ]
          }
        }
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  [Hook] hooks.json not found at {json_path}, using fallback rules only")
            return
        except json.JSONDecodeError as e:
            print(f"⚠️  [Hook] hooks.json parse error: {e}, using fallback rules only")
            return

        hooks_data = data.get("hooks", {})
        loaded = 0

        for event_name, rule_list in hooks_data.items():
            # 跳过未知事件
            try:
                event = HookEvent(event_name)
            except ValueError:
                print(f"⚠️  [Hook] Unknown event '{event_name}', skipping")
                continue

            if not isinstance(rule_list, list):
                continue

            for rule_def in rule_list:
                if not isinstance(rule_def, dict):
                    continue

                # 解析 action
                try:
                    action = HookAction(rule_def.get("action", "log"))
                except ValueError:
                    action = HookAction.LOG

                # 解析 conditions
                conditions = []
                for c in rule_def.get("conditions", []):
                    if isinstance(c, dict):
                        conditions.append(Condition(
                            field=c.get("field", ""),
                            operator=c.get("operator", "contains"),
                            pattern=c.get("pattern", ""),
                        ))

                rule = Rule(
                    name=rule_def.get("name", f"rule_{loaded}"),
                    event=event,
                    action=action,
                    matcher=rule_def.get("matcher"),
                    conditions=conditions,
                    message=rule_def.get("message", ""),
                    enabled=rule_def.get("enabled", True),
                )
                self._rules.append(rule)
                loaded += 1

        print(f"✅ [Hook] Loaded {loaded} rules from {json_path}")

    # ── 规则评估 ──────────────────────────────────────────────

    def evaluate(
        self,
        event: HookEvent,
        tool_name: Optional[str] = None,
        context: dict = None,
    ) -> HookResult:
        """
        评估所有匹配规则，返回 HookResult。

        借鉴 rule_engine.py evaluate_rules()：
        - blocking_rules 优先于 warning_rules
        - ALL conditions 必须满足（AND 语义）
        - 多条 blocking_rules 的 message 合并

        借鉴 Codex hook_runtime.rs run_pre_tool_use_hooks()：
        - 返回 should_block + block_reason
        """
        if context is None:
            context = {}

        blocking: list[Rule] = []
        warning:  list[Rule] = []

        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.event != event:
                continue
            if rule.matcher and tool_name and rule.matcher != tool_name:
                continue
            if self._rule_matches(rule, tool_name or "", context):
                if rule.action == HookAction.BLOCK:
                    blocking.append(rule)
                elif rule.action == HookAction.WARN:
                    warning.append(rule)
                elif rule.action == HookAction.LOG:
                    print(f"📋 [Hook][{rule.name}] {rule.message}")

        if blocking:
            messages = [f"[{r.name}] {r.message}" for r in blocking]
            block_reason = "\n".join(messages)
            print(f"🚫 [Hook] Blocked by: {[r.name for r in blocking]}")
            return HookResult(
                should_block=True,
                block_reason=block_reason,
            )

        if warning:
            messages = [f"[{r.name}] {r.message}" for r in warning]
            warn_text = "\n".join(messages)
            print(f"⚠️  [Hook] Warning: {[r.name for r in warning]}")
            return HookResult(
                should_block=False,
                warn_message=warn_text,
            )

        return HookResult()

    # ── 内部匹配逻辑 ──────────────────────────────────────────

    def _rule_matches(self, rule: Rule, tool_name: str, context: dict) -> bool:
        """
        检查规则是否匹配。

        修复：
        - 无条件 BLOCK → False（太危险，防误杀）
        - 无条件 WARN/LOG → True（仅通知，安全）
        - 有条件 → ALL 条件必须满足
        """
        if not rule.conditions:
            # 无条件 BLOCK 太危险，不允许
            # 无条件 WARN/LOG 安全，允许触发
            return rule.action != HookAction.BLOCK

        for condition in rule.conditions:
            if not self._check_condition(condition, tool_name, context):
                return False
        return True

    def _check_condition(
        self, condition: Condition, tool_name: str, context: dict
    ) -> bool:
        """
        检查单个条件。
        借鉴 rule_engine.py _check_condition() + _extract_field()。
        """
        value = self._extract_field(condition.field, tool_name, context)
        if value is None:
            return False

        op = condition.operator
        pattern = condition.pattern

        if op == "regex_match":
            return self._regex_match(pattern, value)
        elif op == "contains":
            return pattern in value
        elif op == "equals":
            return pattern == value
        elif op == "not_contains":
            return pattern not in value
        elif op == "starts_with":
            return value.startswith(pattern)
        elif op == "ends_with":
            return value.endswith(pattern)
        else:
            print(f"⚠️  [Hook] Unknown operator '{op}'")
            return False

    def _extract_field(
        self, field: str, tool_name: str, context: dict
    ) -> Optional[str]:
        """
        从 context 提取字段值。
        借鉴 rule_engine.py _extract_field() 的字段映射逻辑。

        context 结构（由 hooked_tools_node 构建）：
        {
          "tool_name": "mcp_execute_bash",
          "command": "...",       # execute_bash 的参数
          "file_path": "...",     # write_file / read_file 的参数
          "content": "...",       # write_file 的参数
          "result": "...",        # PostToolUse 时的工具返回值
        }
        """
        # 直接从 context 取
        if field in context:
            v = context[field]
            return v if isinstance(v, str) else str(v)

        # 特殊字段别名（借鉴 rule_engine.py 的 tool-specific 映射）
        if field == "command":
            return context.get("command", "")
        if field in ("file_path", "path"):
            return context.get("file_path", "")
        if field in ("content", "new_text"):
            return context.get("content", "")

        return None

    def _regex_match(self, pattern: str, text: str) -> bool:
        """借鉴 rule_engine.py _regex_match() + compile_regex LRU 缓存"""
        try:
            return bool(_compile_regex(pattern).search(text))
        except re.error as e:
            print(f"⚠️  [Hook] Invalid regex '{pattern}': {e}")
            return False

    # ── 兼容旧 API（避免 main.py 修改量太大）────────────────

    def run_hooks(
        self,
        event: HookEvent,
        tool_name: Optional[str] = None,
        context: dict = None,
    ) -> Optional[str]:
        """
        兼容旧 API：返回阻断消息字符串或 None。
        新代码应使用 evaluate() 获取完整 HookResult。
        """
        result = self.evaluate(event, tool_name, context)
        return result.block_reason if result.should_block else None