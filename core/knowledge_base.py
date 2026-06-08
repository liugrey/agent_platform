"""
core/knowledge_base.py
Agent 独立知识库：Markdown 预处理、语义切块、Embedding、Qdrant 检索、Reranker 精排。
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source: str
    text: str
    score: float
    rerank_score: Optional[float]
    document_index: int
    chunk_index: int
    chunk_count: int
    original_text: str
    metadata: dict[str, Any]


@dataclass
class MarkdownSection:
    title: str = ""
    level: int = 0
    lines: list[str] = field(default_factory=list)
    children: list["MarkdownSection"] = field(default_factory=list)


class SemanticChunker:
    def __init__(self, settings: Settings, embedding_model: "EmbeddingModel"):
        self.settings = settings
        self.embedding_model = embedding_model

    def split(self, text: str) -> list[str]:
        cleaned = self._clean_text(text)
        if not cleaned:
            return []
        units = self._split_units(cleaned)
        return self._merge_units(units) if units else [cleaned]

    def split_markdown(self, markdown: str) -> list[str]:
        cleaned = self._clean_text(markdown)
        if not cleaned:
            return []
        units = self._split_markdown_units(cleaned)
        if not units:
            units = self._split_units(cleaned)
        return self._merge_units(units) if units else [cleaned]

    def _merge_units(self, units: list[str]) -> list[str]:
        normalized = [unit.strip() for unit in units if unit and unit.strip()]
        if not normalized:
            return []

        embeddings = self.embedding_model.embed_texts(normalized)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0
        previous_vector: Optional[list[float]] = None

        for unit, vector in zip(normalized, embeddings):
            unit_length = len(unit)
            similarity = self._cosine_similarity(previous_vector, vector) if previous_vector else 1.0
            would_exceed = current_length + unit_length > self.settings.chunk_max_chars
            below_minimum = current_length < self.settings.chunk_min_chars
            semantic_continue = similarity >= self.settings.chunk_similarity_threshold

            if current_parts and would_exceed:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = [unit]
                current_length = unit_length
            elif current_parts and not below_minimum and not semantic_continue:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = [unit]
                current_length = unit_length
            else:
                current_parts.append(unit)
                current_length += unit_length

            previous_vector = vector

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @classmethod
    def _split_units(cls, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= 300:
                units.append(paragraph)
                continue
            pieces = re.split(r"(?<=[。！？!?；;.\n])\s*", paragraph)
            units.extend([piece.strip() for piece in pieces if piece.strip()])
        return units

    @classmethod
    def _split_markdown_units(cls, text: str) -> list[str]:
        root = cls._parse_markdown_tree(text)
        units: list[str] = []
        cls._collect_section_units(root, [], units)
        return [unit for unit in units if unit.strip()]

    @classmethod
    def _parse_markdown_tree(cls, text: str) -> MarkdownSection:
        root = MarkdownSection()
        stack: list[MarkdownSection] = [root]
        lines = text.split("\n")
        index = 0

        while index < len(lines):
            line = lines[index].rstrip()
            heading = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section = MarkdownSection(title=title, level=level)
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                stack[-1].children.append(section)
                stack.append(section)
                index += 1
                continue

            if cls._looks_like_markdown_table(lines, index):
                table_lines, index = cls._consume_table(lines, index)
                stack[-1].lines.extend(cls._table_to_natural_language(table_lines))
                continue

            stack[-1].lines.append(line)
            index += 1

        return root

    @classmethod
    def _collect_section_units(
        cls,
        section: MarkdownSection,
        parent_titles: list[str],
        units: list[str],
    ) -> None:
        current_titles = parent_titles + ([section.title] if section.title else [])

        for block in cls._split_section_blocks(section.lines):
            text = "\n".join(block).strip()
            if not text:
                continue
            if current_titles:
                units.append(f"标题层级: {' > '.join(current_titles)}\n{text}")
            else:
                units.append(text)

        for child in section.children:
            cls._collect_section_units(child, current_titles, units)

    @staticmethod
    def _split_section_blocks(lines: list[str]) -> list[list[str]]:
        blocks: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append(stripped)
        if current:
            blocks.append(current)
        return blocks

    @staticmethod
    def _looks_like_markdown_table(lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if "|" not in header:
            return False
        return bool(re.match(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$", separator))

    @staticmethod
    def _consume_table(lines: list[str], start: int) -> tuple[list[str], int]:
        table_lines: list[str] = []
        index = start
        while index < len(lines):
            line = lines[index].rstrip()
            if not line.strip():
                break
            if "|" not in line:
                break
            table_lines.append(line)
            index += 1
        return table_lines, index

    @classmethod
    def _table_to_natural_language(cls, table_lines: list[str]) -> list[str]:
        if len(table_lines) < 2:
            return table_lines

        headers = cls._split_table_row(table_lines[0])
        rows = table_lines[2:] if len(table_lines) > 2 else []
        if not headers or not rows:
            return table_lines

        units: list[str] = []
        for row in rows:
            values = cls._split_table_row(row)
            if not values:
                continue
            pairs = []
            for header, value in zip(headers, values):
                clean_header = header.strip()
                clean_value = value.strip()
                if clean_header or clean_value:
                    pairs.append(f"{clean_header}为{clean_value}")
            if pairs:
                units.append("表格行：" + "；".join(pairs))
            else:
                units.append(row.strip())
        return units or table_lines

    @staticmethod
    def _split_table_row(row: str) -> list[str]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        return [cell for cell in cells if cell]

    @staticmethod
    def _cosine_similarity(a: Optional[list[float]], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        numerator = sum(x * y for x, y in zip(a, b))
        a_norm = math.sqrt(sum(x * x for x in a))
        b_norm = math.sqrt(sum(y * y for y in b))
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return numerator / (a_norm * b_norm)


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            if "bge-m3" in self.model_name.lower():
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
            else:
                from FlagEmbedding import FlagModel

                self._model = FlagModel(self.model_name, use_fp16=False)
        except ImportError as exc:
            raise RuntimeError("缺少 FlagEmbedding 依赖，请先安装 requirements.txt 中的 RAG 依赖。") from exc
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        if "bge-m3" in self.model_name.lower():
            result = model.encode(texts, batch_size=min(16, len(texts)), max_length=8192)
            return [list(vector) for vector in result["dense_vecs"]]
        return [list(vector) for vector in model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []


class RerankerModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError("缺少 FlagEmbedding 依赖，请先安装 requirements.txt 中的 RAG 依赖。") from exc
        self._model = FlagReranker(self.model_name, use_fp16=False)
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        pairs = [[query, document] for document in documents]
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(score) for score in scores]


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_model = EmbeddingModel(settings.embedding_model)
        self.reranker_model = RerankerModel(settings.reranker_model)
        self.chunker = SemanticChunker(settings, self.embedding_model)
        self._qdrant = None

    def upsert_document(
        self,
        *,
        agent_id: str,
        collection_name: str,
        document: dict[str, Any],
        previous_collection_name: Optional[str] = None,
    ) -> dict[str, Any]:
        content = str(document.get("content") or "").strip()
        if not content:
            return {
                "collection_name": collection_name,
                "document_id": str(document.get("document_id") or ""),
                "title": str(document.get("title") or ""),
                "source": str(document.get("source") or ""),
                "chunk_count": 0,
            }

        document_id = str(document.get("document_id") or "").strip() or str(uuid.uuid4())
        source = str(document.get("source") or document_id).strip()
        title = str(document.get("title") or source or document_id).strip()
        metadata = dict(document.get("metadata") or {})

        split_chunks = self.chunker.split_markdown(content)
        if not split_chunks:
            split_chunks = [content]
        chunk_count = len(split_chunks)

        client = self._get_qdrant()
        vectors = self.embedding_model.embed_texts(split_chunks)
        self._ensure_collection(client, collection_name, len(vectors[0]))

        self._delete_document_vectors(
            client=client,
            collection_name=collection_name,
            agent_id=agent_id,
            document_id=document_id,
        )
        if previous_collection_name and previous_collection_name != collection_name:
            self._delete_document_vectors(
                client=client,
                collection_name=previous_collection_name,
                agent_id=agent_id,
                document_id=document_id,
            )

        from qdrant_client.http import models

        points = []
        for chunk_index, (chunk_text, vector) in enumerate(zip(split_chunks, vectors)):
            payload = {
                "agent_id": agent_id,
                "collection_name": collection_name,
                "document_id": document_id,
                "title": title,
                "source": source,
                "text": chunk_text,
                "original_text": content,
                "metadata": metadata,
                "document_index": 0,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info(
            "Knowledge upserted | agent=%s collection=%s document=%s chunks=%s",
            agent_id,
            collection_name,
            document_id,
            len(points),
        )
        return {
            "collection_name": collection_name,
            "document_id": document_id,
            "title": title,
            "source": source,
            "chunk_count": len(points),
        }

    def ingest_documents(
        self,
        *,
        agent_id: str,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        chunk_count = 0
        chunk_counts_by_document: dict[str, int] = {}
        for document in documents:
            result = self.upsert_document(
                agent_id=agent_id,
                collection_name=collection_name,
                document=document,
            )
            if result["document_id"]:
                chunk_counts_by_document[result["document_id"]] = result["chunk_count"]
            chunk_count += result["chunk_count"]

        return {
            "collection_name": collection_name,
            "document_count": len(documents),
            "chunk_count": chunk_count,
            "chunk_counts_by_document": chunk_counts_by_document,
        }

    def search(
        self,
        *,
        collection_name: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        query_vector = self.embedding_model.embed_query(query)
        if not query_vector:
            return []

        client = self._get_qdrant()
        self._ensure_collection(client, collection_name, len(query_vector))
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        retrieved = [
            RetrievedChunk(
                chunk_id=str(item.id),
                document_id=str((item.payload or {}).get("document_id", "")),
                source=str((item.payload or {}).get("source", "unknown")),
                text=str((item.payload or {}).get("text", "")),
                score=float(item.score),
                rerank_score=None,
                document_index=int((item.payload or {}).get("document_index", 0)),
                chunk_index=int((item.payload or {}).get("chunk_index", 0)),
                chunk_count=int((item.payload or {}).get("chunk_count", 1)),
                original_text=str((item.payload or {}).get("original_text", "")),
                metadata=dict((item.payload or {}).get("metadata") or {}),
            )
            for item in results
        ]
        if not retrieved:
            return []

        rerank_scores = self.reranker_model.rerank(query, [item.text for item in retrieved])
        reranked = []
        for item, rerank_score in zip(retrieved, rerank_scores):
            item.rerank_score = rerank_score
            reranked.append(item)

        reranked.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else item.score,
            reverse=True,
        )
        return reranked[: self.settings.rerank_top_n]

    def build_context(self, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return ""
        lines = ["以下是可引用的知识库片段，请仅在有帮助时使用："]
        for index, source in enumerate(sources, start=1):
            lines.append(f"[来源{index}] source={source.source}")
            lines.append(source.text)
        return "\n".join(lines)

    def _delete_document_vectors(self, *, client, collection_name: str, agent_id: str, document_id: str) -> None:
        if not collection_name or not self._collection_exists(client, collection_name):
            return

        from qdrant_client.http import models

        client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(key="agent_id", match=models.MatchValue(value=agent_id)),
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                ]
            ),
            wait=True,
        )

    def _get_qdrant(self):
        if self._qdrant is not None:
            return self._qdrant
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("缺少 qdrant-client 依赖，请先安装 requirements.txt 中的 RAG 依赖。") from exc

        if self.settings.qdrant_url:
            self._qdrant = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
        else:
            self._qdrant = QdrantClient(path=self.settings.qdrant_path)
        return self._qdrant

    @staticmethod
    def _collection_exists(client, collection_name: str) -> bool:
        collections = client.get_collections().collections
        existing_names = {collection.name for collection in collections}
        return collection_name in existing_names

    def _ensure_collection(self, client, collection_name: str, vector_size: int):
        from qdrant_client.http import models

        if self._collection_exists(client, collection_name):
            return
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
