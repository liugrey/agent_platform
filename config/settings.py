"""
config/settings.py
统一读取 .env 中的配置，并做必要的类型转换。
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """所有配置项及默认值。"""

    llm_provider: str = "openai"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    bailian_api_key: str = ""
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_model: str = "qwen3.6-plus"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "./data/qdrant"
    qdrant_collection_prefix: str = "agent_"
    retrieval_top_k: int = 8
    rerank_top_n: int = 4
    chunk_max_chars: int = 800
    chunk_min_chars: int = 200
    chunk_similarity_threshold: float = 0.72

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    db_path: str = "./data/agent.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

    @property
    def model_name(self) -> str:
        mapping = {
            "openai": self.openai_model,
            "bailian": self.bailian_model,
            "anthropic": self.anthropic_model,
            "ollama": self.ollama_model,
        }
        return mapping.get(self.llm_provider, self.openai_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
