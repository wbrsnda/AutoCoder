# AutoCoder 🤖

> 基于 LangGraph 的双 Agent 代码协作系统 | A LangGraph-based Dual-Agent Code Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 项目亮点

AutoCoder 是一个**本地化的多 Agent 代码协作系统**，借鉴 [Claude Code](https://www.anthropic.com/claude-code) 和 [OpenAI Codex](https://openai.com/codex) 的核心设计思想，采用 **Architect-Coder 双角色架构**：

- 🏛️ **Architect Agent**：负责任务规划与拆解
- 💻 **Coder Agent**：负责工具调用与代码执行
- 🔧 **MCP Server**：通过 Model Context Protocol 协议封装文件系统、Shell、代码搜索等工具
- 🛡️ **Hook 安全引擎**：声明式 JSON 规则系统，对危险操作三级防护（block / warn / log）
- 🧠 **历史压缩**：渐进式摘要 + 长消息截断，避免 token 浪费
- ⚡ **本地优先**：支持 Ollama、vLLM 等本地 LLM，也兼容 OpenAI API

---

## 🏗️ 架构图
┌─────────────────────────────────────────────┐
│ User Input │
└──────────────────┬──────────────────────────┘
▼
┌──────────────────┐
│ Architect │ ← 规划 / 拆解 / 决策
│ (high temp 0.1) │
└──────────┬───────┘
│ DELEGATE TO CODER
▼
┌──────────────────┐
│ Coder │ ← 工具调用 / 执行
│ (zero temp 0.0) │
└──────────┬───────┘
│ tool_calls
▼
┌──────────────────┐ ┌─────────────┐
│ Hook Engine │ ───→│ MCP Server │
│ (pre/post check) │ │ (7 tools) │
└──────────┬───────┘ └─────────────┘
│ tool results
▼
┌──────────────────┐
│ Budget Control │ ← 死循环检测 / 调用上限
└──────────┬───────┘
│
└──→ Loop back to Coder / Architect

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Ollama（推荐）或任意 OpenAI 兼容的 LLM 服务

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/wbrsnda/AutoCoder.git
cd AutoCoder

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 LLM 配置
# 可使用本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
ARCHITECT_MODEL=gemma4
CODER_MODEL=gemma4

# 或使用 OpenAI
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=sk-xxx
# ARCHITECT_MODEL=gpt-4o
# CODER_MODEL=gpt-4o-mini

# 工作目录（Agent 只能操作此目录）
WORKSPACE_DIR=./workspace

#运行
python -m autocoder.main


#使用示例
🚀 AutoCoder v4 Starting...
✅ Ready. Workspace: ./workspace

🧑 You: 列出工作目录有什么文件，告诉我每个文件的作用

🏛️  Architect:
DELEGATE TO CODER: Read files using mcp_list_dir to explore the workspace.

💻 Coder:
🛠️  Tools: ['mcp_list_dir']

📋 Result: [DIR] src, [FILE] config.py, [FILE] utils.py

🏛️  Architect:
DELEGATE TO CODER: Read files config.py, utils.py simultaneously.

💻 Coder:
🛠️  Tools: ['mcp_read_file', 'mcp_read_file']

🏛️  Architect:
工作目录包含 2 个文件和 1 个目录：
- config.py: 项目全局配置...
- utils.py: 工具函数库...

AWAITING USER INPUT
