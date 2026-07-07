"""
工具参数强类型 Schema - 参数校验与归一化。
从 langchain BaseTool 自动提取（兼容 pydantic v1/v2），运行时校验类型和必填字段。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import inspect

try:
    from pydantic_core import PydanticUndefined
except ImportError:  # pydantic v1 环境
    class PydanticUndefined:  # sentinel
        pass


@dataclass
class ParamSpec:
    name: str
    type_name: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class ToolSchema:
    name: str
    description: str
    params: List[ParamSpec] = field(default_factory=list)

    def validate(self, args: Dict[str, Any]) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """
        校验并归一化参数。返回 (是否合法, 错误消息, 归一化后的 args)。

        ★ 关键保护：如果 schema 推断失败（params 为空），直接放行，
          绝不能把所有参数当"未知参数"删掉。
        """
        normalized = dict(args or {})

        if not self.params:
            return True, None, normalized

        # 1. 必填检查 + 默认值填充
        for p in self.params:
            if p.required and p.name not in normalized:
                return False, f"Missing required param '{p.name}' ({p.type_name})", normalized
            if p.name not in normalized and p.default is not None:
                normalized[p.name] = p.default

        # 2. 类型宽松校验/强制转换（兼容 LLM 输出字符串数字等）
        for p in self.params:
            if p.name not in normalized:
                continue
            val = normalized[p.name]
            if val is None and not p.required:
                continue
            expected = p.type_name.lower()
            if expected == "int":
                try:
                    normalized[p.name] = int(val)
                except (ValueError, TypeError):
                    return False, f"Param '{p.name}' must be int, got {type(val).__name__}", normalized
            elif expected == "float":
                try:
                    normalized[p.name] = float(val)
                except (ValueError, TypeError):
                    return False, f"Param '{p.name}' must be float", normalized
            elif expected == "bool":
                if isinstance(val, str):
                    normalized[p.name] = val.lower() in ("true", "1", "yes")
                else:
                    normalized[p.name] = bool(val)
            elif expected == "str":
                if not isinstance(val, str):
                    normalized[p.name] = str(val)

        # 3. 剔除未知参数（防止 LLM 幻觉参数打爆底层工具）
        known = {p.name for p in self.params}
        for k in list(normalized.keys()):
            if k not in known:
                normalized.pop(k, None)

        return True, None, normalized


def infer_schema_from_langchain_tool(tool_obj) -> ToolSchema:
    """从 langchain @tool 装饰的工具自动生成 ToolSchema。"""
    name = getattr(tool_obj, "name", tool_obj.__class__.__name__)
    description = (getattr(tool_obj, "description", "") or "").strip()

    params: List[ParamSpec] = []
    args_schema = getattr(tool_obj, "args_schema", None)

    # pydantic v2
    if args_schema is not None and hasattr(args_schema, "model_fields"):
        for fname, finfo in args_schema.model_fields.items():
            required = finfo.is_required()
            default = finfo.default
            if default is PydanticUndefined or isinstance(default, type(PydanticUndefined)):
                default = None
            params.append(ParamSpec(
                name=fname,
                type_name=_annotation_to_str(finfo.annotation),
                required=required,
                default=None if required else default,
                description=getattr(finfo, "description", "") or "",
            ))
    # pydantic v1
    elif args_schema is not None and hasattr(args_schema, "__fields__"):
        for fname, finfo in args_schema.__fields__.items():
            required = bool(getattr(finfo, "required", True))
            default = getattr(finfo, "default", None)
            params.append(ParamSpec(
                name=fname,
                type_name=_annotation_to_str(getattr(finfo, "outer_type_", None)),
                required=required,
                default=None if required else default,
            ))
    else:
        # fallback: 函数签名（async @tool 在 .coroutine 上）
        func = getattr(tool_obj, "func", None) or getattr(tool_obj, "coroutine", None)
        if func:
            try:
                sig = inspect.signature(func)
                for pname, p in sig.parameters.items():
                    if pname == "self":
                        continue
                    required = p.default is inspect.Parameter.empty
                    params.append(ParamSpec(
                        name=pname,
                        type_name=_annotation_to_str(p.annotation),
                        required=required,
                        default=None if required else p.default,
                    ))
            except (ValueError, TypeError):
                pass

    return ToolSchema(name=name, description=description, params=params)


def _annotation_to_str(ann) -> str:
    if ann is None or ann is inspect.Parameter.empty:
        return "any"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann).replace("typing.", "")