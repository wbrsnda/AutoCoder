from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class MemorySource(str, Enum):
    STAGE_ONE = "stage_one"
    AD_HOC = "ad_hoc"
    CONSOLIDATED = "consolidated"


@dataclass
class MemoryCitationEntry:
    path: str
    line_start: int
    line_end: int
    note: str

    def to_string(self) -> str:
        return f"{self.path}:{self.line_start}-{self.line_end}|note=[{self.note}]"

    @classmethod
    def parse(cls, line: str) -> Optional["MemoryCitationEntry"]:
        line = line.strip()
        if not line or "|note=[" not in line:
            return None
        try:
            location, note_part = line.rsplit("|note=[", 1)
            note = note_part.rstrip("]").strip()
            path, line_range = location.rsplit(":", 1)
            start, end = line_range.split("-")
            return cls(
                path=path.strip(),
                line_start=int(start),
                line_end=int(end),
                note=note,
            )
        except Exception:
            return None


@dataclass
class MemoryCitation:
    entries: List[MemoryCitationEntry] = field(default_factory=list)
    rollout_ids: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.entries and not self.rollout_ids


@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    content: str = ""
    raw_memory: str = ""
    rollout_summary: str = ""
    slug: str = ""

    source: MemorySource = MemorySource.STAGE_ONE

    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    related_files: List[str] = field(default_factory=list)

    turn_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None

    rollout_path: Optional[str] = None
    cwd: str = "unknown"
    source_updated_at: Optional[datetime] = None

    is_consolidated: bool = False
    consolidated_from: List[str] = field(default_factory=list)
    consolidation_watermark: Optional[int] = None

    expires_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and datetime.now() > self.expires_at

    def to_context_fragment(self, require_citation: bool = True) -> str:
        header = f"[Memory ID: {self.id}]"
        if self.related_files:
            header += f" [Files: {', '.join(self.related_files)}]"
        citation = ""
        if require_citation:
            citation = f"\n[Cite: <rollout_ids>\n{self.id}\n</rollout_ids>]"
        body = self.rollout_summary or self.content
        return f"{header}\n{body}{citation}\n"


@dataclass
class MemoriesConfig:
    generate_memories: bool = True
    use_memories: bool = True

    max_memories_per_turn: int = 4
    max_memory_age_days: int = 30
    max_unused_days: int = 30

    enable_phase2: bool = True
    use_vector_search: bool = False

    # Codex-like startup extraction
    auto_startup_pipeline: bool = True
    max_rollouts_per_startup: int = 8
    max_rollout_age_days: int = 30
    min_rollout_idle_hours: int = 0

    # model hints
    extract_model: Optional[str] = None
    consolidation_model: Optional[str] = None

    max_raw_memories_for_consolidation: int = 12
    startup_concurrency: int = 4