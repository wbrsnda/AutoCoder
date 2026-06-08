# autocoder/rag/retriever.py
"""
Web Search 工具（当前 RAG 的实际实现）。

注意：这是 Web Search，不是本地向量库 RAG。
本地 RAG（ChromaDB + Embedding）在 requirements.txt 中有依赖
但尚未实现，是后续扩展点。
"""
from langchain_core.tools import tool

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    _DDGS_AVAILABLE = False


@tool
async def rag_search(query: str) -> str:
    """
    Search the web for programming knowledge, API documentation, or error solutions.
    Use this when you are uncertain about a library API, framework usage, or error message.
    """
    if not _DDGS_AVAILABLE:
        return (
            "Web search unavailable: 'duckduckgo-search' package not installed.\n"
            "Install with: pip install duckduckgo-search\n"
            "Falling back to internal knowledge only."
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return f"No web results found for: {query}"

        snippets = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body  = r.get("body", "No content")
            href  = r.get("href", "")
            snippets.append(f"[{i}] {title}\nURL: {href}\n{body}")

        return "\n\n".join(snippets)

    except Exception as e:
        return (
            f"Web search failed: {e}\n"
            f"This may be due to network restrictions or proxy settings.\n"
            f"Falling back to internal knowledge only."
        )