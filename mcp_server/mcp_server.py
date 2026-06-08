import os
import re
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AutoCoder-Engine")

WORKSPACE_DIR = Path(os.getenv("AUTOCODER_WORKSPACE", os.getcwd())).resolve()


def _resolve(path: str) -> Path:
    target = (WORKSPACE_DIR / path).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR)):
        raise ValueError(f"Access denied: Path {path} is outside workspace {WORKSPACE_DIR}")
    return target


@mcp.tool()
def list_dir(directory: str = ".") -> str:
    """List directory contents."""
    try:
        target = _resolve(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"

        items = []
        for p in target.iterdir():
            if p.name.startswith('.') or p.name in ['node_modules', '__pycache__', 'venv']:
                continue
            items.append(f"[{'DIR' if p.is_dir() else 'FILE'}] {p.name}")
        return "\n".join(items) if items else "Empty directory."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def read_file(file_path: str, start_line: int = 1, end_line: int = None) -> str:
    """Read file with line numbers."""
    try:
        target = _resolve(file_path)
        with open(target, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        end = min(end_line, len(lines)) if end_line else len(lines)
        return "".join(f"{i+1:4d} | {lines[i]}" for i in range(start_line-1, end))
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
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            results.append(f"{path.relative_to(WORKSPACE_DIR)}:{i}: {line.strip()}")
                            if len(results) > 100:
                                return "\n".join(results) + "\n...[Too many results]"
            except UnicodeDecodeError:
                pass
    except re.error as e:
        return f"Regex Compilation Error: {e}"
    return "\n".join(results) if results else "No matches found."


@mcp.tool()
def execute_bash(command: str, timeout: int = 60) -> str:
    """Execute a shell command safely."""
    if re.search(r"^\s*(vim|nano|top|less|more|htop)", command):
        return "Error: Interactive commands blocked."

    try:
        res = subprocess.run(
            command, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=timeout
        )
        out = f"Exit Code: {res.returncode}\n"
        if res.stdout:
            out += f"--- STDOUT ---\n{res.stdout[:2000]}\n"
        if res.stderr:
            out += f"--- STDERR ---\n{res.stderr[:2000]}\n"
        return out
    except Exception as e:
        return f"Command execution failed: {e}"


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """Write content to a file (creates parent dirs)."""
    try:
        target = _resolve(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return f"Success: Written {file_path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def apply_patch(file_path: str, original: str, replacement: str) -> str:
    """Apply a targeted patch: replace an exact string in a file."""
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