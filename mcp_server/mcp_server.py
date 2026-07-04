import os
import re
import subprocess
import platform
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AutoCoder-Engine")

WORKSPACE_DIR = Path(os.getenv("AUTOCODER_WORKSPACE", os.getcwd())).resolve()

# 借鉴 Codex _sandbox.py：3 级沙箱
SANDBOX_MODE = os.getenv("AUTOCODER_SANDBOX", "workspace_write")


def _resolve(path: str) -> Path:
    """解析路径并做沙箱检查。"""
    target = (WORKSPACE_DIR / path).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR)):
        raise ValueError(f"Access denied: Path {path} is outside workspace {WORKSPACE_DIR}")
    return target


def _check_write_permission() -> str | None:
    """检查写权限。返回错误信息或 None（允许）。"""
    if SANDBOX_MODE == "read_only":
        return "Error: Sandbox is in read-only mode. Write operations are disabled."
    return None


@mcp.tool()
def list_dir(directory: str = ".") -> str:
    """List directory contents."""
    try:
        target = _resolve(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"

        items = []
        for p in target.iterdir():
            if p.name.startswith('.') or p.name in ['node_modules', '__pycache__', 'venv', '.git']:
                continue
            items.append(f"[{'DIR' if p.is_dir() else 'FILE'}] {p.name}")
        return "\n".join(sorted(items)) if items else "Empty directory."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def read_file(file_path: str, start_line: int = 1, end_line: int = None) -> str:
    """Read file with line numbers."""
    try:
        target = _resolve(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        with open(target, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        end = min(end_line, len(lines)) if end_line else len(lines)
        start = max(1, start_line)
        return "".join(f"{i+1:4d} | {lines[i]}" for i in range(start-1, end))
    except UnicodeDecodeError:
        return f"Error: {file_path} is not a text file (binary content)"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def search_files(regex: str, file_pattern: str = "*.*") -> str:
    """Search for regex in files."""
    results = []
    try:
        pattern = re.compile(regex)
        for path in WORKSPACE_DIR.rglob(file_pattern):
            if any(part.startswith('.') for part in path.parts) or not path.is_file():
                continue
            if path.name in ['node_modules', '__pycache__', 'venv']:
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            results.append(f"{path.relative_to(WORKSPACE_DIR)}:{i}: {line.strip()}")
                            if len(results) > 100:
                                return "\n".join(results) + "\n...[Too many results, showing first 100]"
            except (UnicodeDecodeError, PermissionError):
                pass
    except re.error as e:
        return f"Regex Compilation Error: {e}"
    return "\n".join(results) if results else "No matches found."


@mcp.tool()
def execute_bash(command: str, timeout: int = 60) -> str:
    """Execute a shell command safely."""
    if re.search(r"^\s*(vim|nano|top|less|more|htop)", command):
        return "Error: Interactive commands blocked. Use read_file/write_file instead."

    if SANDBOX_MODE == "read_only":
        write_patterns = r"(^|\s)(rm|del|move|mv|cp|mkdir|rmdir|touch|>|>>|tee)\s"
        if re.search(write_patterns, command):
            return "Error: Sandbox is in read-only mode. Destructive/write commands are disabled."

    if platform.system() == "Windows":
        shell_executable = None
    else:
        shell_executable = "/bin/bash"

    try:
        res = subprocess.run(
            command, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=timeout,
            executable=shell_executable,
        )
        out = f"Exit Code: {res.returncode}\n"
        if res.stdout:
            out += f"--- STDOUT ---\n{res.stdout[:3000]}\n"
        if res.stderr:
            out += f"--- STDERR ---\n{res.stderr[:2000]}\n"
        return out
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Command execution failed: {e}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """
    OVERWRITE-write a file (creates parent dirs).
    ⚠️  This REPLACES existing file content. For appending, use append_file instead.
    """
    write_err = _check_write_permission()
    if write_err:
        return write_err
    try:
        target = _resolve(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f"Success: Written (OVERWRITE) {file_path} ({len(content)} bytes, {content.count(chr(10))+1} lines)"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def append_file(file_path: str, content: str, add_newline: bool = True) -> str:
    """
    APPEND content to a file (creates the file if not exists).
    Use this when you want to add content WITHOUT losing existing content.

    Args:
        file_path: Target file path.
        content: Text to append.
        add_newline: If True (default), ensure a newline separator before appending.
    """
    write_err = _check_write_permission()
    if write_err:
        return write_err
    try:
        target = _resolve(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        existed = target.exists()
        existing_size = target.stat().st_size if existed else 0

        # 保证追加前有换行分隔
        prefix = ""
        if add_newline and existed and existing_size > 0:
            with open(target, "rb") as f:
                f.seek(-1, os.SEEK_END)
                last_char = f.read(1)
            if last_char != b"\n":
                prefix = "\n"

        with open(target, "a", encoding="utf-8") as f:
            f.write(prefix + content)

        new_size = target.stat().st_size
        return (
            f"Success: Appended to {file_path} "
            f"(+{len(content) + len(prefix)} bytes, total now {new_size} bytes)"
        )
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def apply_patch(file_path: str, original: str, replacement: str) -> str:
    """Apply a targeted patch: replace an exact string in a file."""
    write_err = _check_write_permission()
    if write_err:
        return write_err
    try:
        target = _resolve(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"

        content = target.read_text(encoding="utf-8")

        if original not in content:
            preview = content[:300].replace("\n", "↵")
            return (
                f"Error: Original text not found in '{file_path}'.\n"
                f"File preview (first 300 chars):\n{preview}\n"
                f"Tip: Use read_file to inspect the exact content first."
            )

        count = content.count(original)
        if count > 1:
            return (
                f"Error: Original text appears {count} times in '{file_path}'. "
                f"Provide more context to make the match unique."
            )

        new_content = content.replace(original, replacement, 1)
        target.write_text(new_content, encoding="utf-8")

        old_lines = original.count("\n")
        new_lines = replacement.count("\n")
        return f"Success: Patched '{file_path}' (-{old_lines} lines / +{new_lines} lines)"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def delete_file(file_path: str) -> str:
    """Delete a file in the workspace."""
    write_err = _check_write_permission()
    if write_err:
        return write_err
    try:
        target = _resolve(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: {file_path} is not a file."

        target.unlink()
        return f"Success: Deleted file {file_path}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run(transport='stdio')