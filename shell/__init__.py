# autocoder/shell/__init__.py
from .detect import ShellType, detect_shell_type, get_platform_commands

__all__ = ["ShellType", "detect_shell_type", "get_platform_commands"]