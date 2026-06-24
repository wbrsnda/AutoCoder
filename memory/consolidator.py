"""
整合 raw_memories.md → MEMORY.md + memory_summary.md
对齐 Codex: 合并旧记忆 + 整合后清空 raw（遗忘原始记录）。
"""
from __future__ import annotations

from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from autocoder.memory.workspace import MemoryWorkspace


CONSOLIDATE_PROMPT = """你是记忆整合代理。

你会收到两部分输入：
1. [现有长期记忆] - 之前已经提炼的事实（可能为空）
2. [新增原始记录] - 最近几轮对话的原始记录

任务：把两者**合并**成一份更新后的长期记忆。

合并规则（重要）：
1. 保留[现有长期记忆]中仍然有效的事实。
2. 从[新增原始记录]中提炼新的事实加进去。
3. **去重**：相同的事实只保留一条。
4. **删除过时/矛盾**：如果新记录与旧记忆矛盾，以新记录为准，删掉旧的。
5. **删除无价值条目**：一次性查询、临时工具调用、系统协议讨论 → 不保留。

提取重点：
- 用户偏好（语言、技术栈、工作习惯）
- 工作目录的关键事实
- 重要决策
- 用户明确要求记住的事

输出格式（严格 Markdown，控制在 2000 字以内）：

# 长期记忆

## 用户偏好
- ...

## 工作目录事实
- ...

## 重要决策
- ...

## 其他
- ...

某节为空就省略。简洁、具体、可操作。不要输出原始对话，只输出提炼后的事实。
"""

SUMMARY_PROMPT = """你是记忆摘要代理。

基于下面的长期记忆文档，生成一份**极简摘要**（≤300字），
作为下次会话注入 system prompt 的上下文。
只保留最关键、跨会话最有用的信息。
输出纯文本，不用 Markdown 标题。
"""

# 防止 MEMORY.md 无限膨胀的硬上限
MAX_MEMORY_MD_CHARS = 8000


class MemoryConsolidator:
    def __init__(self, workspace: MemoryWorkspace, llm):
        self.workspace = workspace
        self.llm = llm

    async def consolidate(self) -> bool:
        raw = self.workspace.read_file("raw_memories.md").strip()
        if not raw or len(raw) < 200:
            return False

        existing = self.workspace.read_file("MEMORY.md").strip()

        try:
            # Step 1: 合并旧记忆 + 新原始记录 → 新 MEMORY.md
            merge_input = (
                f"[现有长期记忆]\n{existing or '(暂无)'}\n\n"
                f"[新增原始记录]\n{raw[-15000:]}"
            )
            res = await self.llm.ainvoke([
                SystemMessage(content=CONSOLIDATE_PROMPT),
                HumanMessage(content=merge_input),
            ])
            memory_md = str(res.content).strip()
            if memory_md.startswith("```"):
                memory_md = memory_md.strip("`").lstrip("markdown").lstrip("\n")

            # 硬上限保护
            if len(memory_md) > MAX_MEMORY_MD_CHARS:
                memory_md = memory_md[:MAX_MEMORY_MD_CHARS] + "\n\n[...truncated]"

            self.workspace.write_file("MEMORY.md", memory_md)

            # Step 2: 生成 summary
            res2 = await self.llm.ainvoke([
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(content=memory_md),
            ])
            summary = str(res2.content).strip()
            self.workspace.write_file("memory_summary.md", summary)

            # Step 3: ★ 遗忘机制 - 归档并清空 raw_memories.md
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.workspace.write_file(
                f"rollout_summaries/archive-{ts}.md",
                f"# Archived raw memories @ {ts}\n\n{raw}"
            )
            self.workspace.write_file("raw_memories.md", "")  # 清空

            # Step 4: 清理过老的归档（保留最近 5 个）
            self._prune_archives(keep=5)

            self.workspace.commit_all(f"consolidate + forget @ {ts}")
            print(f"🧠 [Memory] Consolidated → MEMORY.md ({len(memory_md)}b), "
                  f"summary ({len(summary)}b), raw cleared.")
            return True
        except Exception as e:
            print(f"⚠️  [Memory] Consolidation failed: {e}")
            return False

    def _prune_archives(self, keep: int = 5) -> None:
        """只保留最近 N 个归档文件，其余删除（时间衰减遗忘）"""
        try:
            archive_dir = self.workspace.rollout_summaries_dir
            archives = sorted(
                archive_dir.glob("archive-*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in archives[keep:]:
                old.unlink()
                print(f"🗑️  [Memory] Pruned old archive: {old.name}")
        except Exception:
            pass