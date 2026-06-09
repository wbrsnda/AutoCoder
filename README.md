# AutoCoder 🤖

> 基于 LangGraph 的确定性双 Agent 代码协作系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

借鉴 [Claude Code](https://www.anthropic.com/claude-code) 与 [OpenAI Codex](https://openai.com/codex) 的工程化设计，本地优先、零数据外泄。

---

## ✨ 核心特性

- **三阶段状态机**：Architect 规划 → Coder 执行 → Reporter 解析，物理阻断工具循环
- **确定性代码分析**：Python AST 解析真实类/函数/行号，告别 LLM 幻觉
- **企业级安全**：PreToolUse 拦截 `rm -rf` / `eval()` / `pickle.load()`，三级沙箱隔离
- **Turn 隔离**：每轮工具结果独立作用域，杜绝上下文污染
- **本地优先**：支持 Ollama / vLLM / LM Studio，兼容 OpenAI API

---

## 🏗️ 架构

```
User Input
    ↓
🏛️ Architect  ──→ 规划 / 确认 / 最终回答（不调工具）
    ↓ DELEGATE
💻 Coder      ──→ 单次工具调用（绑定 tools）
    ↓ tool_calls
🛡️ Hook Engine ──→ 危险操作拦截 / 安全扫描
    ↓
🔧 MCP Server ──→ 7 个沙箱工具
    ↓ ToolMessage
📋 Reporter   ──→ AST 解析 + 结构化报告（不绑定 tools）
    ↓
🏛️ Architect  ──→ 基于报告组织回答
```

---

## 🚀 快速开始

```bash
# 1. 安装
git clone https://github.com/yourusername/AutoCoder.git
cd AutoCoder
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置
cp .env.example .env
# 编辑 .env

# 3. 运行
python -m autocoder.main
```

### .env 配置

```env
# Ollama 本地（推荐）
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
ARCHITECT_MODEL=qwen2.5-coder:7b
CODER_MODEL=qwen2.5-coder:7b

# 或 OpenAI
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=sk-xxx
# ARCHITECT_MODEL=gpt-4o
# CODER_MODEL=gpt-4o-mini

# 工作目录（沙箱）
WORKSPACE_DIR=./workspace
AUTOCODER_SANDBOX=workspace_write   # read_only | workspace_write | full_access
```

---

## 🎯 使用示例

### 代码分析（AST 驱动，非 LLM 幻觉）

```text
🧑 You: 分析 train.py 的核心逻辑

📋 Reporter (AST):
- 类: L102 VideoMAERepCountDataset_Density, L231 StrongDensityTransformer
- 函数: L297 train(), L269 evaluate()
- 训练循环: L316-L357 (zero_grad, backward, step)
- 保存: L377 torch.save

🏛️ Architect:
该文件包含训练入口 train()，核心模型是 StrongDensityTransformer
（基于 TransformerEncoder），使用 Adam + MSELoss 优化...
```

### 安全删除（PreToolUse 确认）

```text
🧑 You: 删除 temp.log

🏛️ Architect:
是否确认删除该文件？请回复 YES 或 NO

🧑 You: YES

⚠️ [Hook] Warning: File deletion requested
✅ Success: Deleted temp.log
```

---

## 🛡️ 安全机制

### Hook 规则 (`plugins/hooks.json`)

```json
{
  "PreToolUse": [
    {
      "name": "block-rm-rf",
      "matcher": "mcp_execute_bash",
      "action": "block",
      "conditions": [
        {"field": "command", "operator": "regex_match", "pattern": "rm\\s+-rf"}
      ]
    },
    {
      "name": "warn-eval",
      "matcher": "mcp_write_file",
      "action": "warn",
      "conditions": [
        {"field": "content", "operator": "regex_match", "pattern": "(?<![a-zA-Z0-9_.])eval\\("}
      ]
    }
  ]
}
```

支持 `block` / `warn` / `log` 三级响应，覆盖 `eval` / `pickle` / `os.system` / `shell=True` / TLS 禁用 / 硬编码密钥等 15+ 规则。

---

## 🔧 设计对比

| 问题 | 传统方案 | AutoCoder |
|------|---------|-----------|
| 工具死循环 | prompt 提醒 | **架构阻断**：Reporter 不绑定 tools |
| 代码幻觉 | LLM 自由总结 | **AST 解析**：真实行号 + 类名 |
| 上下文污染 | 扫描历史消息 | **Turn 隔离**：仅当前轮工具结果 |
| 安全滞后 | 事后警告 | **PreToolUse 拦截** |

---

## 📁 项目结构

```
autocoder/
├── agent/
│   ├── state_machine.py    # LangGraph 三阶段状态机
│   └── prompts.py          # System Prompts
├── memory/compress.py      # 历史压缩
├── orchestrator/
│   ├── hook_engine.py      # 安全规则引擎
│   └── security_patterns.py
├── mcp_server/mcp_server.py # MCP 工具服务（7 tools + 沙箱）
├── plugins/hooks.json      # 声明式安全规则
└── main.py
```

---

## 📝 License

MIT License

---

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 状态机框架
- [MCP](https://modelcontextprotocol.io) - 工具协议标准
- [Claude Code](https://www.anthropic.com/claude-code) & [OpenAI Codex](https://openai.com/codex) - 设计理念参考