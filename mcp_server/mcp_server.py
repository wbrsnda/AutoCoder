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


@mcp.tool()
def write_files(files: list[dict]) -> str:
    """
    Batch create/overwrite multiple files at once.
    Args:
        files: List of {"file_path": "...", "content": "..."} dicts
    Returns per-file results.
    """
    write_err = _check_write_permission()
    if write_err:
        return write_err
    if not files:
        return "Error: files list is empty"

    ok, fail = [], []
    for entry in files:
        fp = entry.get("file_path", "")
        content = entry.get("content", "")
        if not fp:
            fail.append(f"  FAIL  missing file_path: {entry}")
            continue
        try:
            target = _resolve(fp)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            ok.append(f"  OK    {fp} ({len(content)} bytes)")
        except Exception as e:
            fail.append(f"  FAIL  {fp}: {e}")

    return f"Batch write: {len(ok)} ok, {len(fail)} failed\n" + "\n".join(ok + fail)


# ═══════════════════════════════════════════════════════
# Batch 1: Project development tools
# ═══════════════════════════════════════════════════════

@mcp.tool()
def find_files(pattern: str = "*.*", directory: str = ".", max_depth: int = None) -> str:
    """
    Find files by glob pattern (e.g. '*.py', 'test_*.py', '**/*.json').
    Use this instead of list_dir when you need to search recursively or filter by name.
    Args:
        pattern: Glob pattern (supports ** for recursive matching). Default: "*.*"
        directory: Starting directory. Default: "." (workspace root)
        max_depth: Optional max directory depth (None = unlimited)
    """
    try:
        target = _resolve(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: {directory} is not a directory"

        matches = []
        if max_depth is not None and max_depth < 0:
            return "Error: max_depth must be >= 0 or None"

        for p in sorted(target.rglob(pattern)):
            if not p.is_file():
                continue
            if any(part.startswith('.') for part in p.parts if part not in ('.', '..')):
                continue
            if any(part in ('node_modules', '__pycache__', 'venv', '.git') for part in p.parts):
                continue
            rel = p.relative_to(target)
            depth = len(rel.parts) - 1
            if max_depth is not None and depth > max_depth:
                continue
            size = p.stat().st_size
            matches.append((str(rel).replace("\\", "/"), size))

        if not matches:
            return f"No files matching '{pattern}' found in {directory}"

        lines = [f"Found {len(matches)} file(s) matching '{pattern}' in {directory}:"]
        for rel_path, size in matches[:100]:
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            lines.append(f"  {rel_path} ({size_str})")

        if len(matches) > 100:
            lines.append(f"  ... (+{len(matches) - 100} more files, use a more specific pattern)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def create_directory(path: str) -> str:
    """
    Create a directory (and any missing parent directories).
    Safe: does nothing if directory already exists.
    """
    write_err = _check_write_permission()
    if write_err:
        return write_err
    try:
        target = _resolve(path)
        existed = target.exists()
        target.mkdir(parents=True, exist_ok=True)
        verb = "Already exists" if existed else "Created"
        return f"Success: {verb} directory: {path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def move_file(source: str, destination: str) -> str:
    """
    Move or rename a file/directory within the workspace.
    Both source and destination must be within workspace boundaries.
    """
    write_err = _check_write_permission()
    if write_err:
        return write_err
    try:
        src = _resolve(source)
        dst = _resolve(destination)

        if not src.exists():
            return f"Error: Source not found: {source}"

        # 如果目标目录不存在，自动创建
        dst.parent.mkdir(parents=True, exist_ok=True)

        src.rename(dst)
        return f"Success: Moved '{source}' → '{destination}'"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def move_files(sources: list[str], destination_dir: str) -> str:
    """
    Batch move multiple files into a target directory.
    Args:
        sources: List of file paths to move
        destination_dir: Target directory (created if missing)
    Returns a summary with per-file results.
    """
    write_err = _check_write_permission()
    if write_err:
        return write_err
    if not sources:
        return "Error: sources list is empty"

    try:
        dst_dir = _resolve(destination_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        ok, fail, skipped = [], [], []
        for src_path in sources:
            try:
                src = _resolve(src_path)
                if not src.exists():
                    fail.append(f"  FAIL  {src_path}: not found")
                    continue
                dst = dst_dir / src.name
                if dst.exists():
                    skipped.append(f"  SKIP  {src_path}: already exists at {destination_dir}/{src.name}")
                    continue
                src.rename(dst)
                ok.append(f"  OK    {src_path} → {destination_dir}/{src.name}")
            except Exception as e:
                fail.append(f"  FAIL  {src_path}: {e}")

        parts = [f"Batch move: {len(ok)} ok, {len(fail)} failed, {len(skipped)} skipped"]
        parts.extend(ok)
        parts.extend(fail)
        parts.extend(skipped)
        return "\n".join(parts)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def git_status() -> str:
    """
    Show the working tree status (git status --short).
    Returns a human-readable summary of staged, unstaged, and untracked changes.
    """
    if not (WORKSPACE_DIR / ".git").exists():
        return "Not a git repository. Use 'git init' in the workspace to initialize one."

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return (
                f"Git error (exit code {result.returncode}):\n"
                f"{result.stderr.strip()[:500]}\n\n"
                f"Hint: Ensure the workspace is a git repository (git init)."
            )

        output = result.stdout.strip()
        if not output:
            return "Working tree clean. No changes to commit."

        lines = output.split("\n")
        staged = [l for l in lines if l[0] != " "]
        unstaged = [l for l in lines if l[0] == " " and l[1] != "?"]
        untracked = [l for l in lines if l.startswith("??")]

        parts = ["### Git Status"]
        if staged:
            parts.append(f"\nStaged for commit ({len(staged)}):")
            parts.extend(f"  {l}" for l in staged[:30])
        if unstaged:
            parts.append(f"\nModified (not staged) ({len(unstaged)}):")
            parts.extend(f"  {l}" for l in unstaged[:30])
        if untracked:
            parts.append(f"\nUntracked ({len(untracked)}):")
            parts.extend(f"  {l}" for l in untracked[:30])

        total = len(lines)
        if total > 30:
            parts.append(f"\n... ({total - 30} more entries, showing first 30)")
        return "\n".join(parts)
    except FileNotFoundError:
        return "Error: git is not installed or not available on PATH."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def git_diff(file_path: str = "", staged: bool = False) -> str:
    """
    Show git diff for a specific file or all changed files.
    Args:
        file_path: Optional path to diff (leave empty for all changes)
        staged: If True, show staged changes (git diff --staged). Default: False (unstaged)
    """
    if not (WORKSPACE_DIR / ".git").exists():
        return "Not a git repository. Use 'git init' in the workspace to initialize one."

    try:
        args = ["git", "diff"]
        if staged:
            args.append("--staged")
        if file_path:
            args.append("--")
            args.append(file_path)

        result = subprocess.run(
            args,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return (
                f"Git diff error (exit code {result.returncode}):\n"
                f"{result.stderr.strip()[:500]}\n\n"
                f"Hint: Ensure the workspace is a git repository."
            )

        output = result.stdout.strip()
        if not output:
            scope = file_path if file_path else "working tree"
            kind = "staged" if staged else "unstaged"
            return f"No {kind} changes in {scope}."

        # 限制输出长度，避免 token 爆炸
        if len(output) > 8000:
            output = output[:8000] + "\n\n[...diff truncated to 8000 chars]"
        return output
    except FileNotFoundError:
        return "Error: git is not installed or not available on PATH."
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run(transport='stdio')