# autocoder/shell/detect.py
"""
平台 Shell 检测 (借鉴 Codex shell_detect.rs)

shell_detect.rs 的核心思路：
- 先精确匹配完整路径名
- 再递归匹配 file_stem（去掉路径前缀和扩展名）
- 枚举覆盖所有已知 shell 变体
"""
import os
import sys
from enum import Enum


class ShellType(str, Enum):
    ZSH = "zsh"
    BASH = "bash"
    POWERSHELL = "powershell"
    SH = "sh"
    CMD = "cmd"


# 借鉴 shell_detect.rs 的精确匹配表
_SHELL_MAP = {
    "zsh":        ShellType.ZSH,
    "bash":       ShellType.BASH,
    "sh":         ShellType.SH,
    "cmd":        ShellType.CMD,
    "pwsh":       ShellType.POWERSHELL,
    "powershell": ShellType.POWERSHELL,
}


def detect_shell_type() -> ShellType:
    """
    检测当前平台的 Shell 类型。
    借鉴 Codex shell_detect.rs：
    - Windows → CMD
    - Unix → 读 $SHELL 环境变量，取 file_stem 后查表
    - 未知 → 降级到 SH
    """
    if sys.platform == "win32":
        return ShellType.CMD

    shell_path = os.environ.get("SHELL", "/bin/bash")
    # 取 file_stem："/usr/local/bin/bash" → "bash"
    stem = os.path.basename(shell_path).lower()
    return _SHELL_MAP.get(stem, ShellType.SH)


def get_platform_commands() -> dict[str, str]:
    """
    返回当前平台的命令映射。
    借鉴 Codex derive_exec_args()。
    """
    if sys.platform == "win32":
        return {
            "list_files":  "dir",
            "delete_file": "del /f /q",
            "delete_dir":  "rmdir /s /q",
            "python":      "python",
            "pip":         "pip",
        }
    return {
        "list_files":  "ls -la",
        "delete_file": "rm -f",
        "delete_dir":  "rm -rf",
        "python":      "python3",
        "pip":         "pip3",
    }