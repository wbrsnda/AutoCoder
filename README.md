# AutoCoder v7 🤖

> 基于 LangGraph + MCP 的本地双 Agent 代码协作系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)](https://modelcontextprotocol.io)

借鉴 Claude Code 与 OpenAI Codex 的工程化设计，本地优先、零数据外泄。

---

## ✨ 当前能力

| 能力 | 说明 |
|------|------|
| **Architect-Coder 双 Agent** | Architect 规划+审查，Coder 执行+写码，物理阻断角色越界 |
| **15 个 MCP 工具** | 文件读写/搜索/批量操作/移动删除/Git/Shell，完整沙箱隔离 |
| **Web Search** | Bing + DuckDuckGo 双引擎，0.5s 返回标题+URL+摘要 |
| **PlannerGuard 确定性执行** | 弱模型补偿，中文自然语言 delegation 自动转 tool_call |
| **Harness 6 层执行闭环** | Schema校验→权限检查→去重→PreHook→执行→自愈→审计 |
| **上下文跟踪 (FileTracker)** | 已读去重/修改检测/stale 标记，防止重复读取 |
| **记忆系统 (Memory)** | 对齐 Codex，每 3 turn 自动摘要压缩 |
| **Token 管理** | 32K 上下文窗口，70% 软压缩 / 85% 硬限制 |
| **15 条安全规则** | rm -rf 拦截 / eval/pickle/subprocess 注入检测 / 硬编码密钥警告 |
| **三级沙箱** | read_only / workspace_write / full_access |

---

## 🏗️ 架构

```
User Input
    ↓
🏛️ Architect (gemma4:32k)  ── 规划 → 搜索 → 审查 ── 不写代码
    ↓ DELEGATE TO CODER
💻 Coder (gemma4:32k)       ── 执行工具 → 写代码 → 返回报告
    ↓ tool_calls
🛡️ Harness 闭环             ── Schema → 权限 → Hook → 执行 → 自愈 → 审计
    ↓
🔧 MCP Server               ── 15 个沙箱工具 + Web Search
    ↓ ToolMessage
📋 Coder Report (确定性)     ── 结构化结果，不传完整文件内容
    ↓
🏛️ Architect                ── 读文件审查 → 修复 → 完成
```

---

## 🚀 快速开始

```bash
# 1. 环境
conda create -n autocoder python=3.11 && conda activate autocoder
pip install -r requirements.txt

# 2. 模型 (Ollama)
ollama pull gemma4
ollama create gemma4:32k -f modelfile  # num_ctx=32768

# 3. 配置 .env
cp .env.example autocoder/.env

# 4. 运行
python -m autocoder.main
```

### .env 配置

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
ARCHITECT_MODEL=gemma4:32k
CODER_MODEL=gemma4:32k
MODEL_CONTEXT_WINDOW=32768

WORKSPACE_DIR=./workspace
AUTOCODER_SANDBOX=workspace_write

# 代理（可选，用于 DDG 搜索）
PROXY=http://127.0.0.1:7890
```

---

## 🎯 使用示例

### 写代码

```text
🧑 You: 写一个番茄钟，页面要模拟星空闪烁特效

🏛️ Architect:
  → rag_search "css starry night animation"
  → DELEGATE TO CODER: Use mcp_write_files to create index.html, style.css, script.js

💻 Coder:
  → mcp_write_files: 3 files, 6KB

🏛️ Architect:
  → mcp_read_file 审查代码
  → DELEGATE TO CODER: Fix style.css use mcp_apply_patch ...

🏛️ Architect: 完成！浏览器打开 index.html 即可看到星空番茄钟
```

### 搜索查资料

```text
🧑 You: 搜索纳斯达克

🏛️ Architect:
  → rag_search "纳斯达克"
  → 返回: Nasdaq官网 / 雪球 / Investing.com 的标题+URL+摘要
```

### 代码审查

```text
🧑 You: 分析项目结构

🏛️ Architect:
  → mcp_list_dir → 列出所有目录和文件
  → mcp_read_file → 逐个读取 Python 文件
  → AST 解析：类/函数/行号（非 LLM 幻觉）
```

---

## 🔧 工具清单 (15 + Web Search)

| 分类 | 工具 |
|------|------|
| 发现 | `list_dir` `read_file` `find_files` `search_files` |
| 创建/编辑 | `write_file` `append_file` `write_files` `apply_patch` `create_directory` |
| 重组 | `move_file` `move_files` `delete_file` |
| Git | `git_status` `git_diff` |
| 执行 | `execute_bash` |
| Web | `rag_search` (Bing 0.5s + DDG 后备，返回标题+URL+摘要) |
| 记忆 | `memories_search` `memories_read` `memories_list` `add_ad_hoc_note` |

---

## 🛡️ 安全机制 (15 条 Hook 规则)

| 规则 | 动作 |
|------|------|
| `rm -rf` / 格式化磁盘 / vim/nano | **block** |
| `subprocess shell=True` / `os.system()` | **warn** |
| `eval()` / `pickle.load()` / `yaml.load()` | **warn** |
| TLS 禁用 / 硬编码密钥 / `.env` 文件写入 | **warn** |

---

## 📁 项目结构

```
autocoder/
├── agent/
│   ├── state_machine.py      # LangGraph 5 节点状态机
│   └── prompts.py            # Architect + Coder System Prompts
├── harness/
│   ├── invoker.py            # 6 层执行闭环引擎
│   ├── permissions.py        # RBAC 4 级权限
│   ├── gateway.py            # 按需工具暴露（省 token）
│   ├── planner_guard.py      # 弱模型补偿 + 搜索触发 + 语义匹配
│   ├── self_heal.py          # 错误自愈建议
│   ├── audit.py / telemetry.py
│   └── schema.py
├── rag/
│   └── retriever.py          # Bing + DDG 双引擎搜索
├── context/
│   ├── file_tracker.py       # 文件读缓存 + stale 检测
│   └── token_tracker.py      # Token 计数 + 自动压缩
├── memory/                   # 对齐 Codex 的记忆系统
├── skills/builtin.py         # 内置技能（删除确认等）
├── orchestrator/hook_engine.py
├── plugins/hooks.json        # 15 条安全规则
├── mcp_server/mcp_server.py  # 15 个 MCP 工具
├── utils/config.py
└── main.py
```

---

## 📝 License

MIT
