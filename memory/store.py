from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from autocoder.memory.models import MemoryItem, MemorySource, MemoriesConfig

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class MemoryStore:
    def __init__(self, workspace_dir: Path, config: MemoriesConfig):
        self.config = config
        self.workspace = workspace_dir
        self.db_path = workspace_dir / ".autocoder" / "memories.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._chroma_collection = None
        self._vector_disabled = False
        if config.use_vector_search and HAS_CHROMADB:
            try:
                chroma_path = workspace_dir / ".autocoder" / "chroma"
                chroma_path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(chroma_path))
                self._chroma_collection = client.get_or_create_collection(
                    name="memories",
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"✅ [Memory] ChromaDB initialized at {chroma_path}")
            except Exception as e:
                print(f"⚠️ [Memory] ChromaDB init failed: {e}. Falling back to SQLite.")
                self._chroma_collection = None

        self._init_sqlite()

    def _init_sqlite(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)

            self._ensure_columns(conn, "memories", {
                "raw_memory": "TEXT DEFAULT ''",
                "rollout_summary": "TEXT DEFAULT ''",
                "slug": "TEXT DEFAULT ''",
                "related_files": "TEXT DEFAULT '[]'",
                "turn_id": "TEXT",
                "thread_id": "TEXT",
                "session_id": "TEXT",
                "rollout_path": "TEXT",
                "cwd": "TEXT DEFAULT 'unknown'",
                "source_updated_at": "TEXT",
                "is_consolidated": "INTEGER DEFAULT 0",
                "consolidated_from": "TEXT DEFAULT '[]'",
                "consolidation_watermark": "INTEGER",
                "expires_at": "TEXT",
            })

            conn.execute("""
                CREATE TABLE IF NOT EXISTS rollout_jobs (
                    rollout_path TEXT PRIMARY KEY,
                    session_id TEXT,
                    cwd TEXT DEFAULT 'unknown',
                    file_mtime REAL NOT NULL,
                    last_processed_mtime REAL,
                    stage1_status TEXT DEFAULT 'pending',
                    memory_id TEXT,
                    last_error TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(last_accessed_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source, is_consolidated)"
            )

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                print(f"✅ [Memory] Added missing column: {table}.{name}")

    # ── memory CRUD ──────────────────────────────────────────

    def save_memory(self, memory: MemoryItem) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (
                    id, content, raw_memory, rollout_summary, slug, source,
                    created_at, last_accessed_at, access_count,
                    related_files, turn_id, thread_id, session_id,
                    rollout_path, cwd, source_updated_at,
                    is_consolidated, consolidated_from, consolidation_watermark,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.raw_memory,
                    memory.rollout_summary,
                    memory.slug,
                    memory.source.value,
                    memory.created_at.isoformat(),
                    memory.last_accessed_at.isoformat(),
                    memory.access_count,
                    json.dumps(memory.related_files),
                    memory.turn_id,
                    memory.thread_id,
                    memory.session_id,
                    memory.rollout_path,
                    memory.cwd,
                    memory.source_updated_at.isoformat() if memory.source_updated_at else None,
                    1 if memory.is_consolidated else 0,
                    json.dumps(memory.consolidated_from),
                    memory.consolidation_watermark,
                    memory.expires_at.isoformat() if memory.expires_at else None,
                ),
            )

        if self._chroma_collection and not self._vector_disabled and memory.content:
            try:
                self._chroma_collection.upsert(
                    ids=[memory.id],
                    documents=[memory.content[:2000]],
                    metadatas=[{
                        "source": memory.source.value,
                        "cwd": memory.cwd,
                        "slug": memory.slug,
                    }],
                )
            except Exception as e:
                self._vector_disabled = True
                self._chroma_collection = None
                print(f"⚠️ [Memory] ChromaDB disabled due to upsert error: {e}. Falling back to SQLite.")

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            return self._row_to_memory(row) if row else None

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        if self._chroma_collection and self.config.use_vector_search and not self._vector_disabled:
            try:
                results = self._chroma_collection.query(
                    query_texts=[query],
                    n_results=min(limit, 10),
                )
                ids = results.get("ids", [[]])[0]
                if ids:
                    found = [self.get_memory_by_id(mid) for mid in ids]
                    return [m for m in found if m is not None][:limit]
            except Exception as e:
                self._vector_disabled = True
                self._chroma_collection = None
                print(f"⚠️ [Memory] ChromaDB disabled due to query error: {e}. Falling back to SQLite.")

        return self._search_like(query, limit)

    def _search_like(self, query: str, limit: int) -> List[MemoryItem]:
        q = f"%{query}%"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE (content LIKE ? OR raw_memory LIKE ? OR rollout_summary LIKE ? OR slug LIKE ?)
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY last_accessed_at DESC
                LIMIT ?
                """,
                (q, q, q, q, datetime.now().isoformat(), limit),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def touch(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (datetime.now().isoformat(), memory_id),
            )

    def get_memory_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ── rollout jobs ─────────────────────────────────────────

    def scan_rollout_candidates(
        self,
        rollouts_dir: Path,
        active_session_id: Optional[str],
        limit: int,
        max_age_days: int,
        min_idle_hours: int,
    ) -> List[dict]:
        rollouts_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now().timestamp()
        max_age_ts = now - max_age_days * 86400
        idle_ts = now - min_idle_hours * 3600

        candidates = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for path in sorted(rollouts_dir.glob("*.jsonl")):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue

                mtime = stat.st_mtime
                session_id = path.stem

                if active_session_id and session_id == active_session_id:
                    continue
                if mtime < max_age_ts:
                    continue
                if mtime > idle_ts:
                    continue

                row = conn.execute(
                    "SELECT * FROM rollout_jobs WHERE rollout_path = ?",
                    (str(path),),
                ).fetchone()

                processed = False
                if row is not None:
                    last_processed_mtime = row["last_processed_mtime"]
                    status = row["stage1_status"]
                    if last_processed_mtime is not None and last_processed_mtime >= mtime and status in ("succeeded", "no_output"):
                        processed = True

                conn.execute(
                    """
                    INSERT OR REPLACE INTO rollout_jobs
                    (rollout_path, session_id, cwd, file_mtime, last_processed_mtime, stage1_status, memory_id, last_error, updated_at)
                    VALUES (
                        ?, ?,
                        COALESCE((SELECT cwd FROM rollout_jobs WHERE rollout_path = ?), 'unknown'),
                        ?,
                        COALESCE((SELECT last_processed_mtime FROM rollout_jobs WHERE rollout_path = ?), NULL),
                        COALESCE((SELECT stage1_status FROM rollout_jobs WHERE rollout_path = ?), 'pending'),
                        COALESCE((SELECT memory_id FROM rollout_jobs WHERE rollout_path = ?), NULL),
                        COALESCE((SELECT last_error FROM rollout_jobs WHERE rollout_path = ?), ''),
                        ?
                    )
                    """,
                    (
                        str(path), session_id, str(path),
                        mtime, str(path), str(path), str(path), str(path),
                        datetime.now().isoformat(),
                    ),
                )

                if not processed:
                    candidates.append({
                        "rollout_path": str(path),
                        "session_id": session_id,
                        "file_mtime": mtime,
                    })

        candidates.sort(key=lambda x: x["file_mtime"])
        return candidates[:limit]

    def mark_rollout_processed(
        self,
        rollout_path: str,
        session_id: str,
        file_mtime: float,
        status: str,
        memory_id: Optional[str] = None,
        error: str = "",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rollout_jobs
                (rollout_path, session_id, cwd, file_mtime, last_processed_mtime, stage1_status, memory_id, last_error, updated_at)
                VALUES (
                    ?,
                    ?,
                    COALESCE((SELECT cwd FROM rollout_jobs WHERE rollout_path = ?), 'unknown'),
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    rollout_path,
                    session_id,
                    rollout_path,
                    file_mtime,
                    file_mtime,
                    status,
                    memory_id,
                    error,
                    datetime.now().isoformat(),
                ),
            )

    def mark_rollout_failed(self, rollout_path: str, session_id: str, file_mtime: float, error: str) -> None:
        self.mark_rollout_processed(
            rollout_path=rollout_path,
            session_id=session_id,
            file_mtime=file_mtime,
            status="failed",
            memory_id=None,
            error=error[:500],
        )

    # ── stage1 / stage2 ──────────────────────────────────────

    def prune_stage1_outputs_for_retention(self, max_unused_days: int, batch_size: int = 200) -> int:
        cutoff = (datetime.now() - timedelta(days=max_unused_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id FROM memories
                WHERE source = 'stage_one'
                  AND last_accessed_at < ?
                LIMIT ?
                """,
                (cutoff, batch_size),
            ).fetchall()
            ids = [r[0] for r in rows]
            for mid in ids:
                conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
            return len(ids)

    def get_stage1_for_consolidation(self, limit: int = 12, max_age_days: int = 30) -> List[MemoryItem]:
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE source = 'stage_one'
                  AND is_consolidated = 0
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY COALESCE(source_updated_at, created_at) ASC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def mark_stage1_consolidated(self, memory_ids: List[str], watermark: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for mid in memory_ids:
                conn.execute(
                    """
                    UPDATE memories
                    SET is_consolidated = 1,
                        consolidation_watermark = ?
                    WHERE id = ?
                    """,
                    (watermark, mid),
                )

    def get_consolidated_memories(self) -> List[MemoryItem]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE source = 'consolidated' OR is_consolidated = 1
                ORDER BY created_at DESC
                LIMIT 30
                """
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

    # ── helpers ──────────────────────────────────────────────

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        if row is None:
            raise ValueError("row is None")
        keys = set(row.keys())
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            raw_memory=row["raw_memory"] if "raw_memory" in keys else "",
            rollout_summary=row["rollout_summary"] if "rollout_summary" in keys else "",
            slug=row["slug"] if "slug" in keys else "",
            source=MemorySource(row["source"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
            access_count=row["access_count"],
            related_files=json.loads(row["related_files"] or "[]") if "related_files" in keys else [],
            turn_id=row["turn_id"] if "turn_id" in keys else None,
            thread_id=row["thread_id"] if "thread_id" in keys else None,
            session_id=row["session_id"] if "session_id" in keys else None,
            rollout_path=row["rollout_path"] if "rollout_path" in keys else None,
            cwd=row["cwd"] if "cwd" in keys and row["cwd"] else "unknown",
            source_updated_at=datetime.fromisoformat(row["source_updated_at"]) if "source_updated_at" in keys and row["source_updated_at"] else None,
            is_consolidated=bool(row["is_consolidated"]) if "is_consolidated" in keys else False,
            consolidated_from=json.loads(row["consolidated_from"] or "[]") if "consolidated_from" in keys else [],
            consolidation_watermark=row["consolidation_watermark"] if "consolidation_watermark" in keys else None,
            expires_at=datetime.fromisoformat(row["expires_at"]) if "expires_at" in keys and row["expires_at"] else None,
        )