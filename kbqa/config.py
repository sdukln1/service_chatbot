from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    import os

    return os.environ.get(name, default)


@dataclass
class Config:
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    db_dir: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "kbqa"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 5
    score_threshold: float = 0.30

    use_hybrid: bool = True
    hybrid_candidates: int = 20
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60

    use_rerank: bool = True
    rerank_backend: str = "llm"  # llm | off
    rerank_top_n: int = 3
    rerank_threshold: float = 3.0

    cache_enabled: bool = True
    cache_threshold: float = 0.92
    cache_max_size: int = 256
    fallback_answer: str = "抱歉，服务暂时出现异常，请稍后重试，或联系人工客服。"
    feedback_dir: Path = PROJECT_ROOT / "data" / "feedback"

    agent_enabled: bool = True
    agent_max_steps: int = 3

    api_base: str = _env("KBQA_API_BASE", "http://localhost:8000/v1")
    api_key: str = _env("KBQA_API_KEY", "EMPTY")
    llm_models: list = field(
        default_factory=lambda: [
            m.strip()
            for m in _env(
                "KBQA_LLM_MODELS", "GLM-5.3-Flash,DeepSeek-V4-Flash-0731,GLM-5.3"
            ).split(",")
            if m.strip()
        ]
    )
    llm_model: str = _env("KBQA_LLM_MODEL", "GLM-5.3-Flash")
    verify_ssl: bool = False
    temperature: float = 0.2
    max_tokens: int = 1024

    server_host: str = "0.0.0.0"
    server_port: int = int(_env("KBQA_SERVER_PORT", "8080"))

    models_dir: Path = PROJECT_ROOT / "data" / "models"

    # LangFuse 可观测性（Phase 5，可选）：不配置时自动禁用，所有调用失败不阻塞主链路
    langfuse_enabled: bool = _env("KBQA_LANGFUSE_ENABLED", "false").lower() == "true"
    langfuse_host: str = _env("KBQA_LANGFUSE_HOST", "")
    langfuse_public_key: str = _env("KBQA_LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = _env("KBQA_LANGFUSE_SECRET_KEY", "")


def resolve_model(name: str) -> str:
    """本地模型目录中存在同名目录时优先使用本地路径（离线环境）。"""
    local = CONFIG.models_dir / name.split("/")[-1]
    if local.exists():
        return str(local)
    return name


CONFIG = Config()
