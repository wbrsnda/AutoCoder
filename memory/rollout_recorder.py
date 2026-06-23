from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class RolloutRecorder:
    def __init__(self, workspace_dir: Path, session_id: str | None = None):
        self.workspace_dir = workspace_dir
        self.rollouts_dir = workspace_dir / ".autocoder" / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or self._new_session_id()
        self.rollout_path = self.rollouts_dir / f"{self.session_id}.jsonl"

        if not self.rollout_path.exists():
            self._append_json({
                "type": "session_meta",
                "session_id": self.session_id,
                "started_at": datetime.now().isoformat(),
                "cwd": str(workspace_dir),
            })

    def _new_session_id(self) -> str:
        return datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "-" + uuid.uuid4().hex[:8]

    def append_turn(
        self,
        turn_id: str,
        user_input: str,
        assistant_response: str,
        tool_records: list[dict[str, Any]],
        cwd: str,
        file_stats: dict[str, Any] | None = None,
    ) -> None:
        self._append_json({
            "type": "turn",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "turn_id": turn_id,
            "cwd": cwd,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "tool_records": tool_records,
            "file_stats": file_stats or {},
        })

    def _append_json(self, obj: dict) -> None:
        with self.rollout_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")