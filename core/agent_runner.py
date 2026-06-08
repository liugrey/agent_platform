"""
core/agent_runner.py
根据 Agent 配置执行对话，并接入每个 Agent 的独立知识库。
"""

from __future__ import annotations

import logging
import uuid
from typing import Generator, Optional

from config.settings import Settings
from core.knowledge_base import KnowledgeBase, RetrievedChunk
from core.llm_client import LLMClient
from storage.db import Database

logger = logging.getLogger(__name__)


def _build_llm_for_agent(agent: dict, base_settings: Settings) -> LLMClient:
    provider = agent.get("provider") or base_settings.llm_provider
    model = agent.get("model") or base_settings.model_name

    overrides = {
        "llm_provider": provider,
        f"{provider}_model": model,
        "openai_api_key": base_settings.openai_api_key,
        "openai_base_url": base_settings.openai_base_url,
        "openai_model": base_settings.openai_model,
        "bailian_api_key": base_settings.bailian_api_key,
        "bailian_base_url": base_settings.bailian_base_url,
        "bailian_model": base_settings.bailian_model,
        "anthropic_api_key": base_settings.anthropic_api_key,
        "anthropic_model": base_settings.anthropic_model,
        "ollama_base_url": base_settings.ollama_base_url,
        "ollama_model": base_settings.ollama_model,
        "embedding_model": base_settings.embedding_model,
        "reranker_model": base_settings.reranker_model,
        "qdrant_url": base_settings.qdrant_url,
        "qdrant_api_key": base_settings.qdrant_api_key,
        "qdrant_path": base_settings.qdrant_path,
        "qdrant_collection_prefix": base_settings.qdrant_collection_prefix,
        "retrieval_top_k": base_settings.retrieval_top_k,
        "rerank_top_n": base_settings.rerank_top_n,
        "chunk_max_chars": base_settings.chunk_max_chars,
        "chunk_min_chars": base_settings.chunk_min_chars,
        "chunk_similarity_threshold": base_settings.chunk_similarity_threshold,
        "host": base_settings.host,
        "port": base_settings.port,
        "debug": base_settings.debug,
        "db_path": base_settings.db_path,
    }
    overrides[f"{provider}_model"] = model
    return LLMClient(Settings(**overrides))


class AgentRunner:
    def __init__(
        self,
        agent: dict,
        db: Database,
        settings: Settings,
    ):
        self.agent = agent
        self.db = db
        self.settings = settings
        self.llm = _build_llm_for_agent(agent, settings)
        self.knowledge_base = KnowledgeBase(settings)

    @property
    def collection_name(self) -> str:
        return self.agent.get("collection_name", "")

    @property
    def system_prompt(self) -> str:
        return self.agent.get("system_prompt", "")

    @property
    def temperature(self) -> float:
        return float(self.agent.get("temperature", 0.7))

    @property
    def max_tokens(self) -> int:
        return int(self.agent.get("max_tokens", 2048))

    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> tuple[str, str, list[RetrievedChunk]]:
        session_id = self._ensure_session(session_id, user_message)
        history = self.db.get_history_for_llm(session_id)
        sources = self._retrieve_sources(user_message)
        history.append({"role": "user", "content": self._build_user_message(user_message, sources)})

        reply = self.llm.chat(
            messages=history,
            system_prompt=self._build_system_prompt(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._save_turn(session_id, user_message, reply)
        logger.info("[Agent:%s] session=%s reply_chars=%s", self.agent["name"], session_id, len(reply))
        return reply, session_id, sources

    def chat_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> tuple[Generator[str, None, None], str, list[RetrievedChunk]]:
        session_id = self._ensure_session(session_id, user_message)
        history = self.db.get_history_for_llm(session_id)
        sources = self._retrieve_sources(user_message)
        history.append({"role": "user", "content": self._build_user_message(user_message, sources)})

        def _gen():
            chunks: list[str] = []
            for chunk in self.llm.chat_stream(
                messages=history,
                system_prompt=self._build_system_prompt(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                chunks.append(chunk)
                yield chunk
            self._save_turn(session_id, user_message, "".join(chunks))

        return _gen(), session_id, sources

    def _retrieve_sources(self, user_message: str) -> list[RetrievedChunk]:
        if not self.collection_name:
            return []
        try:
            return self.knowledge_base.search(
                collection_name=self.collection_name,
                query=user_message,
            )
        except Exception as exc:
            logger.warning(
                "[Agent:%s] knowledge retrieval skipped: %s",
                self.agent.get("name", self.agent.get("id")),
                exc,
            )
            return []

    def _build_system_prompt(self) -> str:
        prompt = self.system_prompt.strip()
        rag_instruction = (
            "如果提供了知识库片段，请优先基于知识库回答；"
            "如果知识库不足以回答，请明确说明。回答时不要编造来源。"
        )
        return f"{prompt}\n\n{rag_instruction}".strip()

    def _build_user_message(self, user_message: str, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return user_message
        context = self.knowledge_base.build_context(sources)
        return f"{context}\n\n用户问题：{user_message}"

    def _ensure_session(self, session_id: Optional[str], title: str) -> str:
        sid = session_id or str(uuid.uuid4())
        session = self.db.get_session(sid)
        if session and session.get("agent_id") not in (None, self.agent["id"]):
            raise ValueError(f"session_id 已属于其他 Agent: {sid}")
        self.db.create_session(sid, title=title[:30], agent_id=self.agent["id"])
        return sid

    def _save_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        self.db.add_message(session_id, "user", user_msg)
        self.db.add_message(session_id, "assistant", assistant_msg)
