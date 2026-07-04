import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
    ARCHITECT_MODEL = os.getenv("ARCHITECT_MODEL", "gemma4")
    CODER_MODEL = os.getenv("CODER_MODEL", "gemma4")
    WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve()
    PROXY = os.getenv("PROXY")
    NO_PROXY = os.getenv("NO_PROXY")

    # ── Token 管理（对齐 Codex）──
    MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "8192"))
    AUTO_COMPACT_TOKEN_RATIO = float(os.getenv("AUTO_COMPACT_TOKEN_RATIO", "0.75"))
    HARD_LIMIT_RATIO = float(os.getenv("HARD_LIMIT_RATIO", "0.90"))
    MAX_TOOL_OUTPUT_TOKENS = int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "2000"))
    KEEP_RECENT_MESSAGES = int(os.getenv("KEEP_RECENT_MESSAGES", "10"))

    # 兼容
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    MAX_GUARD_RETRIES = int(os.getenv("MAX_GUARD_RETRIES", "1"))

    @classmethod
    def apply_proxy(cls):
        no_proxy = cls.NO_PROXY or "localhost,127.0.0.1,::1"
        os.environ["NO_PROXY"] = no_proxy
        if cls.PROXY:
            os.environ["HTTP_PROXY"] = cls.PROXY
            os.environ["HTTPS_PROXY"] = cls.PROXY