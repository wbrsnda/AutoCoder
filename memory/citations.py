from __future__ import annotations

import re
from typing import Optional

from autocoder.memory.models import MemoryCitation, MemoryCitationEntry


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
    seen_ids = set()

    entries_block = extract_block(text, "<citation_entries>", "</citation_entries>")
    if entries_block:
        for line in entries_block.splitlines():
            entry = MemoryCitationEntry.parse(line)
            if entry:
                entries.append(entry)

    for open_tag, close_tag in [
        ("<rollout_ids>", "</rollout_ids>"),
        ("<thread_ids>", "</thread_ids>"),
    ]:
        ids_block = extract_block(text, open_tag, close_tag)
        if ids_block:
            for line in ids_block.splitlines():
                line = line.strip()
                if line and line not in seen_ids:
                    seen_ids.add(line)
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