"""
审计日志 - JSONL 持久化所有工具调用。
支持事后回溯、合规审计、故障排查。
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from autocoder.harness.telemetry import ToolSpan


class AuditLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _current_file(self) -> Path:
        # 按天分文件（跨天运行也正确）
        return self.log_dir / f"audit-{datetime.now().strftime('%Y%m%d')}.jsonl"

    def write(self, span: ToolSpan, extra: Optional[dict] = None) -> None:
        record = span.to_dict()
        record["logged_at"] = datetime.now().isoformat()
        if extra:
            record["extra"] = extra
        try:
            with open(self._current_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"⚠️ [Audit] write failed: {e}")

    def query_recent(self, limit: int = 20) -> list[dict]:
        path = self._current_file()
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out