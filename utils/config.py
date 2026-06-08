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

    @classmethod
    def apply_proxy(cls):
        # ★ 始终设置 NO_PROXY，避免本地 Ollama 请求被系统代理拦截
        no_proxy = cls.NO_PROXY or "localhost,127.0.0.1,::1"
        os.environ["NO_PROXY"] = no_proxy
        if cls.PROXY:
            os.environ["HTTP_PROXY"] = cls.PROXY
            os.environ["HTTPS_PROXY"] = cls.PROXY