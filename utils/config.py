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

    # ── 上下文与安全控制 ──
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    # 当上下文总字符数超过此门槛时触发实时压缩 (12000字符 ≈ 5500 tokens，为8192预留充足安全空间)
    COMPRESS_MAX_CHARS = int(os.getenv("COMPRESS_MAX_CHARS", "12000"))
    MAX_GUARD_RETRIES = int(os.getenv("MAX_GUARD_RETRIES", "1"))

    @classmethod
    def apply_proxy(cls):
        no_proxy = cls.NO_PROXY or "localhost,127.0.0.1,::1"
        os.environ["NO_PROXY"] = no_proxy
        if cls.PROXY:
            os.environ["HTTP_PROXY"] = cls.PROXY
            os.environ["HTTPS_PROXY"] = cls.PROXY