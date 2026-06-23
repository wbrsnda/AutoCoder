from __future__ import annotations

import asyncio
from pathlib import Path

from autocoder.memory.models import MemoriesConfig
from autocoder.memory.store import MemoryStore
from autocoder.memory.extractor import StageOneExtractor
from autocoder.memory.consolidator import PhaseTwoConsolidator


class MemoryStartupPipeline:
    def __init__(
        self,
        workspace_dir: Path,
        config: MemoriesConfig,
        store: MemoryStore,
        extractor: StageOneExtractor,
        consolidator: PhaseTwoConsolidator,
    ):
        self.workspace_dir = workspace_dir
        self.config = config
        self.store = store
        self.extractor = extractor
        self.consolidator = consolidator
        self.rollouts_dir = workspace_dir / ".autocoder" / "rollouts"
        self.rollouts_dir.mkdir(parents=True, exist_ok=True)

    async def run_once(self, active_session_id: str | None = None) -> None:
        if not self.config.auto_startup_pipeline:
            return

        print("🧠 [Memory] Startup pipeline: scanning historical rollouts...")

        pruned = self.store.prune_stage1_outputs_for_retention(
            max_unused_days=self.config.max_unused_days,
            batch_size=200,
        )
        if pruned:
            print(f"🧹 [Memory] Pruned {pruned} stale stage1 memories")

        candidates = self.store.scan_rollout_candidates(
            rollouts_dir=self.rollouts_dir,
            active_session_id=active_session_id,
            limit=self.config.max_rollouts_per_startup,
            max_age_days=self.config.max_rollout_age_days,
            min_idle_hours=self.config.min_rollout_idle_hours,
        )

        if not candidates:
            print("✅ [Memory] Startup pipeline: no rollout candidates")
            return

        print(f"🧠 [Memory] Startup pipeline: {len(candidates)} rollout candidate(s)")

        sem = asyncio.Semaphore(self.config.startup_concurrency)
        extracted_any = False

        async def _handle(candidate: dict):
            nonlocal extracted_any
            rollout_path = Path(candidate["rollout_path"])
            session_id = candidate["session_id"]
            file_mtime = candidate["file_mtime"]

            async with sem:
                try:
                    mem = await self.extractor.extract_from_rollout(
                        rollout_path=rollout_path,
                        session_id=session_id,
                    )
                    if mem is None:
                        self.store.mark_rollout_processed(
                            rollout_path=str(rollout_path),
                            session_id=session_id,
                            file_mtime=file_mtime,
                            status="no_output",
                            memory_id=None,
                        )
                        return
                    extracted_any = True
                    self.store.mark_rollout_processed(
                        rollout_path=str(rollout_path),
                        session_id=session_id,
                        file_mtime=file_mtime,
                        status="succeeded",
                        memory_id=mem.id,
                    )
                except Exception as e:
                    self.store.mark_rollout_failed(
                        rollout_path=str(rollout_path),
                        session_id=session_id,
                        file_mtime=file_mtime,
                        error=str(e),
                    )
                    print(f"⚠️ [Memory] Startup pipeline failed for {rollout_path.name}: {e}")

        await asyncio.gather(*[_handle(c) for c in candidates])

        if extracted_any:
            await self.consolidator.run_consolidation("memory_startup")

        print("✅ [Memory] Startup pipeline complete")