import os
from pathlib import Path
from dotenv import load_dotenv

# 显式加载 autocoder/ 目录下的 .env（load_dotenv() 默认找 CWD 及父目录，找不到子目录）
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)


class Config:
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    ARCHITECT_MODEL = os.getenv("ARCHITECT_MODEL", "gemma4:32k")
    CODER_MODEL = os.getenv("CODER_MODEL", "qwen2.5-coder:32k")
    WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
    PROXY = os.getenv("PROXY")
    NO_PROXY = os.getenv("NO_PROXY")

    # ── Token 管理 ──
    # gemma4:32k 支持 32768 上下文，qwen2.5-coder 也支持 32768
    MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "32768"))
    AUTO_COMPACT_TOKEN_RATIO = float(os.getenv("AUTO_COMPACT_TOKEN_RATIO", "0.70"))
    HARD_LIMIT_RATIO = float(os.getenv("HARD_LIMIT_RATIO", "0.85"))
    MAX_TOOL_OUTPUT_TOKENS = int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "4000"))
    KEEP_RECENT_MESSAGES = int(os.getenv("KEEP_RECENT_MESSAGES", "15"))

    # 兼容
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
    MAX_GUARD_RETRIES = int(os.getenv("MAX_GUARD_RETRIES", "1"))

    @classmethod
    def apply_proxy(cls):
        no_proxy = cls.NO_PROXY or "localhost,127.0.0.1,::1"
        os.environ["NO_PROXY"] = no_proxy
        if cls.PROXY:
            os.environ["HTTP_PROXY"] = cls.PROXY
            os.environ["HTTPS_PROXY"] = cls.PROXY