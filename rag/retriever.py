# autocoder/rag/retriever.py
"""
Web Search 工具 — 双引擎架构，对标 Hermes web_search 质量。

Bing（优先）：国内直连，~0.5s，提取标题+URL+摘要
DuckDuckGo（后备）：直连→代理重试，返回含摘要的完整结果
"""
import os, re, urllib.request, urllib.parse
from langchain_core.tools import tool

# ── Bing: 直连打开器（绕过系统代理） ─────────────────────
_BING_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# ── DuckDuckGo ─────────────────────────────────────────
_DDGS = None
try:
    from ddgs import DDGS; _DDGS = DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS; _DDGS = DDGS
    except ImportError:
        pass

# ── Bing 解析 ───────────────────────────────────────────

_BING_SNIPPET_RE = re.compile(
    r'<(p|div)\s[^>]*\b(?:b_lineclamp|b_caption|b_snippet)[^>]*>'
    r'(.*?)'
    r'</\1>',
    re.I | re.S,
)

_BING_H2_RE = re.compile(
    r'<h2[^>]*>.*?<a\s[^>]*\bhref\s*=\s*"'
    r'(https?://[^"]+)'
    r'"[^>]*>'
    r'(.*?)'
    r'</a>',
    re.I | re.S,
)

_NOISE_DOMAINS = (
    'bing.com', 'microsoft.com', 'go.microsoft',
    'miit.gov', 'beian.', 'gov.cn', 'mps.gov',
)


def _strip_html(text: str) -> str:
    """去 HTML 标签，规范化空白。"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&ensp;|&nbsp;|&#\d+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _search_bing(query: str, max_results: int = 5) -> list[dict]:
    """Bing 搜索，提取标题+URL+摘要。"""
    encoded = urllib.parse.quote(query)
    url = f"https://www.bing.com/search?q={encoded}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with _BING_OPENER.open(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    # 定位结果区
    body_start = 0
    for marker in ('<ol id="b_results"', '<li class="b_algo"'):
        idx = html.find(marker)
        if idx > 0:
            body_start = idx
            break
    if body_start == 0:
        return []

    zone = html[body_start:body_start + 50000]

    # 提取所有 <h2> → 标题+URL
    entries = []  # (href, title)
    seen = set()
    for m in _BING_H2_RE.finditer(zone):
        href = m.group(1)
        title = _strip_html(m.group(2))
        if any(d in href.lower() for d in _NOISE_DOMAINS):
            continue
        if href in seen or len(title) < 8:
            continue
        seen.add(href)
        entries.append((href, title))

    # 提取所有 snippet
    snippets = []
    for m in _BING_SNIPPET_RE.finditer(zone):
        text = _strip_html(m.group(2))
        if 30 < len(text) < 600:
            snippets.append(text)

    # 配对：每对 (标题, snippet) 组成一个结果
    results = []
    for i, (href, title) in enumerate(entries):
        body = snippets[i] if i < len(snippets) else f"Result from {urllib.parse.urlparse(href).netloc}"
        results.append({"title": title[:120], "url": href, "body": body[:400]})
        if len(results) >= max_results:
            break

    return results


# ── DuckDuckGo ─────────────────────────────────────────

def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """DDG 搜索。先直连，再走代理。"""
    if _DDGS is None:
        return []

    def _try():
        with _DDGS(timeout=10) as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    for clear_proxy in (True, False):
        saved = {}
        if clear_proxy:
            for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
                saved[k] = os.environ.pop(k, None)
        try:
            raw = _try()
            if raw:
                return [
                    {"title": r.get("title", "")[:120],
                     "url": r.get("href", ""),
                     "body": r.get("body", "")[:400]}
                    for r in raw
                ]
        except Exception:
            pass
        finally:
            if clear_proxy:
                for k, v in saved.items():
                    if v is not None:
                        os.environ[k] = v
    return []


# ── LangChain Tool ─────────────────────────────────────

@tool
async def rag_search(query: str) -> str:
    """
    Search the web for information, documentation, or answers.
    Returns titles, URLs, and content snippets.

    Tips:
    - Be specific with keywords
    - Include error messages verbatim for debugging searches
    """
    if not query or not query.strip():
        return "Error: search query is empty."

    # Bing 优先（直连，快）
    results = _search_bing(query, max_results=5)

    # DDG 后备（质量更高但可能慢）
    if not results:
        results = _search_ddg(query, max_results=5)

    if not results:
        return (
            f"No results found for: {query}\n\n"
            f"Tips: try more specific keywords, or check network/proxy settings."
        )

    out = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        body = r.get("body", "")
        out.append(f"[{i}] {title}\n    URL: {url}\n    {body}\n")

    return "\n".join(out)
