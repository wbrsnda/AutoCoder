#!/usr/bin/env python3
"""
test_ddgs_debug.py  —— 看清楚 DDG 到底回了什么
"""
import json
import logging
import urllib3
from ddgs import DDGS

# 把 HTTP 对话打出来（能看到状态码、是否302/403/空body）
logging.basicConfig(level=logging.DEBUG)
urllib3.disable_warnings()

QUERY = "2026世界杯赛程"

def main():
    print("🔍 Query:", QUERY)
    print("-" * 60)

    try:
        with DDGS(timeout=15, verify=True) as ddgs:
            results = list(ddgs.text(QUERY, max_results=3))

        print("-" * 60)
        print("📦 results type:", type(results))
        print("📦 results len :", len(results))

        if results:
            for i, r in enumerate(results, 1):
                print(f"\n### [{i}]")
                print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        else:
            print("\n⚠️  结果是空的 —— DDG 大概率拦了请求")
            print("    常见原因：IP被限流 / 网络出口被识别为非浏览器 / 需要Cookie/挑战")
            print("    下一步：要么走代理，要么换搜索后端（见下方建议）")

    except Exception as e:
        print("❌ 异常:", type(e).__name__, e)

if __name__ == "__main__":
    main()