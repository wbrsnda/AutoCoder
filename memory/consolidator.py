"""
整合 raw_memories.md → MEMORY.md + memory_summary.md
"""
from __future__ import annotations

from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from autocoder.memory.workspace import MemoryWorkspace


CONSOLIDATE_PROMPT = """你是记忆整合代理。

任务：阅读下面的原始对话记录，提炼出对未来会话有用的**事实性长期记忆**。

提取重点：
1. 用户偏好（"我喜欢用中文" / "我的项目是 ML 研究"）
2. 工作目录的关键事实（"目录下都是零散脚本不是完整项目"）
3. 重要决策（"我们决定用方案 A"）
4. 用户明确要求记住的事（"记住我喜欢 PyTorch"）

**不要**记录：
- 临时的工具调用细节
- 一次性的查询
- 系统内部协议讨论

输出格式（严格 Markdown）：
长期记忆
用户偏好
...
工作目录事实
...
重要决策
...
其他
...

如果某节没有内容就省略。简洁、具体、可操作。
"""

SUMMARY_PROMPT = """你是记忆摘要代理。

基于下面的长期记忆文档，生成一份**极简的摘要**（≤300字），作为下次会话开始时注入到 system prompt 的上下文。

只保留最关键、跨会话最有用的信息。

输出格式：纯文本，不用 Markdown 标题。
"""


class MemoryConsolidator:
    def __init__(self, workspace: MemoryWorkspace, llm):
        self.workspace = workspace
        self.llm = llm

    async def consolidate(self) -> bool:
        raw = self.workspace.read_file("raw_memories.md").strip()
        if not raw or len(raw) < 200:
            return False

        try:
            # Step 1: 提炼 MEMORY.md
            res = await self.llm.ainvoke([
                SystemMessage(content=CONSOLIDATE_PROMPT),
                HumanMessage(content=raw[-15000:]),  # 只取最近 15K 字符
            ])
            memory_md = str(res.content).strip()
            if memory_md.startswith("```"):
                memory_md = memory_md.strip("`").lstrip("markdown").lstrip("\n")

            self.workspace.write_file("MEMORY.md", memory_md)

            # Step 2: 提炼 summary
            res2 = await self.llm.ainvoke([
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(content=memory_md),
            ])
            summary = str(res2.content).strip()

            self.workspace.write_file("memory_summary.md", summary)

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.workspace.commit_all(f"consolidate memory @ {ts}")
            print(f"🧠 [Memory] Consolidated → MEMORY.md ({len(memory_md)}b) + summary ({len(summary)}b)")
            return True
        except Exception as e:
            print(f"⚠️  [Memory] Consolidation failed: {e}")
            return False