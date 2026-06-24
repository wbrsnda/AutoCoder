"""
Citation 解析（自包含，不依赖已删除的 models.py）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MemoryCitationEntry:
    path: str
    line_start: int
    line_end: int
    note: str

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
            return cls(path=path.strip(), line_start=int(start), line_end=int(end), note=note)
        except Exception:
            return None


@dataclass
class MemoryCitation:
    entries: List[MemoryCitationEntry] = field(default_factory=list)
    rollout_ids: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.entries and not self.rollout_ids


def extract_block(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    if open_tag not in text or close_tag not in text:
        return None
    try:
        _, rest = text.split(open_tag, 1)
        body, _ = rest.split(close_tag, 1)
        return body.strip()
    except ValueError:
        return None


def parse_memory_citation(text: str) -> Optional[MemoryCitation]:
    entries = []
    rollout_ids = []
    seen = set()

    block = extract_block(text, "<citation_entries>", "</citation_entries>")
    if block:
        for line in block.splitlines():
            e = MemoryCitationEntry.parse(line)
            if e:
                entries.append(e)

    for o, c in [("<rollout_ids>", "</rollout_ids>"), ("<thread_ids>", "</thread_ids>")]:
        b = extract_block(text, o, c)
        if b:
            for line in b.splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    rollout_ids.append(line)

    if not entries and not rollout_ids:
        return None
    return MemoryCitation(entries=entries, rollout_ids=rollout_ids)


def strip_citations(text: str) -> str:
    patterns = [
        r"<oai-mem-citation>.*?</oai-mem-citation>",
        r"<citation_entries>.*?</citation_entries>",
        r"<rollout_ids>.*?</rollout_ids>",
        r"<thread_ids>.*?</thread_ids>",
        r"\[Memory ID: .*?\]",
        r"\[Cite:.*?\]",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "", out, flags=re.DOTALL)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()