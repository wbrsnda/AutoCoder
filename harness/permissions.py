"""
权限边界 - RBAC 风格分级权限。
每个工具声明所需最低权限级别，Invoker 运行时强制检查。
"""
from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


class PermissionLevel(IntEnum):
    """权限级别（数值越大权限越高）"""
    READ = 10        # 只读
    WRITE = 20       # 工作区写入（含删除工作区文件——确认由 Architect 流程 + Hook 保证）
    EXECUTE = 30     # shell 执行
    DANGEROUS = 40   # 需显式二次授权的操作（预留给未来的 rollback / 系统级操作）


@dataclass
class PermissionPolicy:
    """当前会话的权限策略"""
    max_level: PermissionLevel = PermissionLevel.WRITE
    dangerous_confirmed_tools: Set[str] = field(default_factory=set)

    @classmethod
    def from_sandbox_mode(cls, sandbox_mode: str) -> "PermissionPolicy":
        mapping = {
            "read_only": PermissionLevel.READ,
            "workspace_write": PermissionLevel.EXECUTE,
            "full_access": PermissionLevel.DANGEROUS,
        }
        return cls(max_level=mapping.get(sandbox_mode, PermissionLevel.EXECUTE))

    def check(self, tool_name: str, required: PermissionLevel) -> tuple[bool, Optional[str]]:
        if required > self.max_level:
            return False, (
                f"Permission denied: '{tool_name}' requires {required.name} "
                f"but session sandbox max is {self.max_level.name}"
            )
        if required == PermissionLevel.DANGEROUS and tool_name not in self.dangerous_confirmed_tools:
            return False, f"'{tool_name}' requires explicit user confirmation (DANGEROUS level)"
        return True, None

    def grant_dangerous(self, tool_name: str) -> None:
        self.dangerous_confirmed_tools.add(tool_name)


# 工具 → 所需权限映射（与当前项目实际工具一一对应）
# 注意：mcp_delete_file 归为 WRITE 而非 DANGEROUS ——
# 因为你现有的 delete 技能流程是 Architect 先向用户确认、再委派删除，
# 若归为 DANGEROUS 会在 workspace_write 沙箱下被永久拒绝，破坏现有能力。
DEFAULT_TOOL_PERMISSIONS: Dict[str, PermissionLevel] = {
    # READ
    "mcp_list_dir": PermissionLevel.READ,
    "mcp_read_file": PermissionLevel.READ,
    "mcp_search_files": PermissionLevel.READ,
    "mcp_find_files": PermissionLevel.READ,
    "mcp_git_status": PermissionLevel.READ,
    "mcp_git_diff": PermissionLevel.READ,
    "memories_search": PermissionLevel.READ,
    "memories_read": PermissionLevel.READ,
    "memories_list": PermissionLevel.READ,
    "rag_search": PermissionLevel.READ,
    # WRITE
    "mcp_write_file": PermissionLevel.WRITE,
    "mcp_append_file": PermissionLevel.WRITE,
    "mcp_apply_patch": PermissionLevel.WRITE,
    "mcp_delete_file": PermissionLevel.WRITE,
    "mcp_write_files": PermissionLevel.WRITE,
    "mcp_create_directory": PermissionLevel.WRITE,
    "mcp_move_file": PermissionLevel.WRITE,
    "mcp_move_files": PermissionLevel.WRITE,
    "add_ad_hoc_note": PermissionLevel.WRITE,
    # EXECUTE
    "mcp_execute_bash": PermissionLevel.EXECUTE,
}