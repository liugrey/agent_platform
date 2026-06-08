"""
api/agents.py
Agent 管理、知识库管理与 Agent 对话接口。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import Field

from api.camel_model import CamelModel
from api.pagination import PaginatedResponse, PaginationQuery
from config.settings import Settings, get_settings
from core.agent_runner import AgentRunner
from core.knowledge_base import KnowledgeBase, RetrievedChunk
from storage.agent_store import AgentStore, ROLE_PRESETS
from storage.db import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])

_agent_store: Optional[AgentStore] = None
_db: Optional[Database] = None
_knowledge_base: Optional[KnowledgeBase] = None


def get_agent_store(settings: Settings = Depends(get_settings)) -> AgentStore:
    global _agent_store
    if _agent_store is None:
        _agent_store = AgentStore(settings.db_path)
    return _agent_store


def get_db(settings: Settings = Depends(get_settings)) -> Database:
    global _db
    if _db is None:
        _db = Database(settings.db_path)
    return _db


def get_knowledge_base(settings: Settings = Depends(get_settings)) -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(settings)
    return _knowledge_base


class CreateAgentRequest(CamelModel):
    name: str = Field(..., description="Agent 唯一名称", min_length=1, max_length=64)
    role_type: str = Field(default="assistant", description="角色类型")
    model: str = Field(..., description="模型名称")
    provider: str = Field(
        default="openai",
        description="LLM provider: openai | bailian | anthropic | ollama",
    )
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示词")
    description: str = Field(default="", description="Agent 描述")
    collection_name: Optional[str] = Field(default=None, description="专属知识库 Collection 名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class UpdateAgentRequest(CamelModel):
    name: Optional[str] = None
    role_type: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    system_prompt: Optional[str] = None
    description: Optional[str] = None
    collection_name: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    is_active: Optional[bool] = None


class AgentChatRequest(CamelModel):
    message: str = Field(..., min_length=1, description="用户消息")
    session_id: Optional[str] = Field(default=None, description="会话 ID，为空时自动创建")


class KnowledgeDocumentUpsertRequest(CamelModel):
    document_id: Optional[str] = Field(default=None, description="原始文档 ID")
    title: Optional[str] = Field(default=None, description="文档标题")
    source: str = Field(..., description="文档来源标识，如文件名、URL 或业务主键")
    collection_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9]+$",
        description="向量库 Collection 名称，仅支持英文和数字",
    )
    content: str = Field(..., min_length=1, description="Markdown 文档内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    model_config = CamelModel.model_config | {
        "json_schema_extra": {
            "example": {
                "documentId": "employee-handbook",
                "title": "员工手册",
                "source": "employee-handbook.md",
                "collectionName": "kb202606",
                "content": "# 员工手册\n\n## 请假制度\n\n| 类型 | 说明 |\n| --- | --- |\n| 年假 | 需提前一天申请 |",
                "metadata": {"department": "hr"},
            }
        }
    }


class KnowledgeSearchRequest(CamelModel):
    query: str = Field(..., min_length=1, description="检索问题")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="向量召回数量")


class KnowledgeSourceResponse(CamelModel):
    chunk_id: str = Field(..., description="知识片段 ID")
    document_id: str = Field(..., description="原始文档 ID")
    source: str = Field(..., description="来源标识")
    text: str = Field(..., description="命中的文本片段")
    original_text: str = Field(..., description="入库前原始文本")
    document_index: int = Field(..., description="原始文档序号")
    chunk_index: int = Field(..., description="当前切块序号")
    chunk_count: int = Field(..., description="原始文档切块总数")
    score: float = Field(..., description="向量召回得分")
    rerank_score: Optional[float] = Field(default=None, description="重排得分")
    metadata: dict[str, Any] = Field(default_factory=dict, description="来源元数据")

    model_config = CamelModel.model_config | {
        "json_schema_extra": {
            "example": {
                "chunkId": "36fd68a8-6d05-4fd1-923a-4bc63be81ab4",
                "documentId": "employee-handbook",
                "source": "employee-handbook.md",
                "text": "公司请假制度要求提前一天提交申请。",
                "originalText": "公司请假制度要求提前一天提交申请。如有特殊情况请联系 HR。",
                "documentIndex": 0,
                "chunkIndex": 0,
                "chunkCount": 3,
                "score": 0.82,
                "rerankScore": 0.97,
                "metadata": {"section": "leave-policy"},
            }
        }
    }


class RolePresetResponse(CamelModel):
    role_type: str
    label: str
    description: str


class RoleListResponse(CamelModel):
    roles: list[RolePresetResponse]
    total: int


class RolePromptResponse(CamelModel):
    role_type: str
    system_prompt: str


class AgentResponse(CamelModel):
    id: str
    name: str
    role_type: str
    label: str
    description: str
    system_prompt: str
    model: str
    provider: str
    collection_name: str
    temperature: float
    max_tokens: int
    is_active: bool
    created_at: str
    updated_at: str
    extra: dict[str, Any] = Field(default_factory=dict)


class AgentListResponse(CamelModel):
    agents: list[AgentResponse]
    total: int


class DeleteResponse(CamelModel):
    message: str


class AgentChatResponse(CamelModel):
    reply: str
    session_id: str
    agent_id: str
    agent_name: str
    model: str
    provider: str
    sources: list[KnowledgeSourceResponse] = Field(default_factory=list)


class KnowledgeIngestResponse(CamelModel):
    agent_id: str
    collection_name: str
    document_id: str
    title: str
    source: str
    chunk_count: int
    updated: bool


class KnowledgeDocumentSummaryResponse(CamelModel):
    agent_id: str
    collection_name: str
    document_id: str
    title: str
    source: str
    chunk_count: int
    created_at: str
    updated_at: str


class KnowledgeDocumentSummaryPageResponse(PaginatedResponse[KnowledgeDocumentSummaryResponse]):
    pass


class KnowledgeDocumentDetailResponse(CamelModel):
    agent_id: str
    collection_name: str
    document_id: str
    title: str
    source: str
    content: str
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class KnowledgeSearchResponse(CamelModel):
    agent_id: str
    collection_name: str
    query: str
    sources: list[KnowledgeSourceResponse]


class AgentSessionSummaryResponse(CamelModel):
    id: str
    agent_id: str
    created_at: str
    title: str
    last_msg: Optional[str] = None


class AgentSessionMessageResponse(CamelModel):
    role: str
    content: str
    created_at: str


class AgentSessionHistoryResponse(CamelModel):
    agent_id: str
    session_id: str
    title: str
    created_at: str
    messages: list[AgentSessionMessageResponse]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_stream(generator) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()
    sentinel = object()

    def _next(it):
        try:
            return next(it)
        except StopIteration:
            return sentinel

    it = iter(generator)
    while True:
        chunk = await loop.run_in_executor(None, _next, it)
        if chunk is sentinel:
            break
        yield chunk


def _to_source_responses(sources: list[RetrievedChunk]) -> list[KnowledgeSourceResponse]:
    return [
        KnowledgeSourceResponse(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            source=source.source,
            text=source.text,
            original_text=source.original_text,
            document_index=source.document_index,
            chunk_index=source.chunk_index,
            chunk_count=source.chunk_count,
            score=source.score,
            rerank_score=source.rerank_score,
            metadata=source.metadata,
        )
        for source in sources
    ]


@router.get("/roles", response_model=RoleListResponse, summary="查询所有预置角色类型")
async def list_roles():
    return RoleListResponse(roles=AgentStore.list_role_presets(), total=len(ROLE_PRESETS))


@router.get(
    "/roles/{roleType}/prompt",
    response_model=RolePromptResponse,
    summary="查看角色的默认系统提示词",
)
async def get_role_preset_prompt(role_type: Annotated[str, Path(alias="roleType")]):
    if role_type not in ROLE_PRESETS:
        raise HTTPException(status_code=404, detail=f"角色类型不存在: {role_type}")
    return RolePromptResponse(role_type=role_type, system_prompt=AgentStore.get_preset_prompt(role_type))


@router.post("", response_model=AgentResponse, summary="创建 Agent", status_code=201)
async def create_agent(
    req: CreateAgentRequest,
    store: AgentStore = Depends(get_agent_store),
):
    if store.get_by_name(req.name):
        raise HTTPException(status_code=409, detail=f"Agent 名称已存在: {req.name}")
    if req.role_type not in ROLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"未知角色类型: {req.role_type}，可选: {list(ROLE_PRESETS.keys())}",
        )

    agent = store.create(
        name=req.name,
        role_type=req.role_type,
        model=req.model,
        provider=req.provider,
        system_prompt=req.system_prompt,
        description=req.description,
        collection_name=req.collection_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return AgentResponse(**agent)


@router.get("", response_model=AgentListResponse, summary="查询 Agent 列表")
async def list_agents(
    role_type: Optional[str] = Query(default=None, alias="roleType"),
    active_only: bool = Query(default=True, alias="activeOnly"),
    limit: int = Query(default=50),
    store: AgentStore = Depends(get_agent_store),
):
    agents = store.list_agents(role_type=role_type, active_only=active_only, limit=limit)
    return AgentListResponse(agents=[AgentResponse(**agent) for agent in agents], total=len(agents))


@router.get("/{agentId}", response_model=AgentResponse, summary="查询单个 Agent")
async def get_agent(agent_id: Annotated[str, Path(alias="agentId")], store: AgentStore = Depends(get_agent_store)):
    if agent_id:
        agent = store.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    return AgentResponse(**agent)


@router.get(
    "/{agentId}/sessions",
    response_model=list[AgentSessionSummaryResponse],
    summary="查询 Agent 对话会话列表",
)
async def list_agent_sessions(
    agent_id: Annotated[str, Path(alias="agentId")],
    limit: int = Query(default=20),
    store: AgentStore = Depends(get_agent_store),
    db: Database = Depends(get_db),
):
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    sessions = db.list_sessions(limit=limit, agent_id=agent_id)
    return [AgentSessionSummaryResponse(**session) for session in sessions]


@router.get(
    "/{agentId}/sessions/{sessionId}",
    response_model=AgentSessionHistoryResponse,
    summary="查询 Agent 某个会话详情",
)
async def get_agent_session_history(
    agent_id: Annotated[str, Path(alias="agentId")],
    session_id: Annotated[str, Path(alias="sessionId")],
    store: AgentStore = Depends(get_agent_store),
    db: Database = Depends(get_db),
):
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")

    session = db.get_session(session_id)
    if not session or session.get("agent_id") != agent_id:
        raise HTTPException(status_code=404, detail=f"Agent 会话不存在: {session_id}")

    messages = db.get_messages(session_id)
    return AgentSessionHistoryResponse(
        agent_id=agent_id,
        session_id=session_id,
        title=session.get("title") or "",
        created_at=session["created_at"],
        messages=[AgentSessionMessageResponse(**message) for message in messages],
    )


@router.put("/{agentId}", response_model=AgentResponse, summary="更新 Agent 配置")
async def update_agent(
    req: UpdateAgentRequest,
    agent_id: Annotated[str, Path(alias="agentId")],
    store: AgentStore = Depends(get_agent_store),
):
    if not store.get(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    if req.name:
        existing = store.get_by_name(req.name)
        if existing and existing["id"] != agent_id:
            raise HTTPException(status_code=409, detail=f"名称已被占用: {req.name}")

    updates = {key: value for key, value in req.model_dump().items() if value is not None}
    agent = store.update(agent_id, **updates)
    return AgentResponse(**agent)


@router.delete("/{agentId}", response_model=DeleteResponse, summary="删除 Agent")
async def delete_agent(agent_id: Annotated[str, Path(alias="agentId")], store: AgentStore = Depends(get_agent_store)):
    if not store.get(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    store.delete(agent_id)
    return DeleteResponse(message=f"Agent {agent_id} 已删除")


@router.post(
    "/{agentId}/knowledge/documents",
    response_model=KnowledgeIngestResponse,
    summary="向 Agent 知识库插入文档",
)
async def ingest_knowledge_documents(
    req: KnowledgeDocumentUpsertRequest,
    agent_id: Annotated[str, Path(alias="agentId")],
    store: AgentStore = Depends(get_agent_store),
    db: Database = Depends(get_db),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
):
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")

    existing_document = (
        db.get_agent_knowledge_document(agent_id=agent_id, document_id=req.document_id)
        if req.document_id
        else None
    )

    try:
        result = knowledge_base.upsert_document(
            agent_id=agent_id,
            collection_name=req.collection_name,
            document=req.model_dump(),
            previous_collection_name=existing_document["collection_name"] if existing_document else None,
        )
        db.upsert_knowledge_documents(
            agent_id=agent_id,
            collection_name=result["collection_name"],
            documents=[
                {
                    **req.model_dump(),
                    "document_id": result["document_id"],
                    "title": result["title"],
                    "source": result["source"],
                }
            ],
            chunk_counts={result["document_id"]: result["chunk_count"]},
        )
    except Exception as exc:
        logger.error("[Agent:%s] ingest failed: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return KnowledgeIngestResponse(
        agent_id=agent_id,
        collection_name=result["collection_name"],
        document_id=result["document_id"],
        title=result["title"],
        source=result["source"],
        chunk_count=result["chunk_count"],
        updated=existing_document is not None,
    )

def _build_knowledge_document_page(
    *,
    agent_id: Optional[str],
    pagination: PaginationQuery,
    store: AgentStore,
    db: Database,
) -> KnowledgeDocumentSummaryPageResponse:
    if not agent_id:
        total = db.count_agent_knowledge_documents(agent_id=None)
        documents = db.list_agent_knowledge_documents(
            agent_id=None,
            limit=pagination.page_size,
            offset=pagination.offset,
        )
        return KnowledgeDocumentSummaryPageResponse.create(
            items=[KnowledgeDocumentSummaryResponse(**document) for document in documents],
            page=pagination.page,
            page_size=pagination.page_size,
            total=total,
        )

    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    total = db.count_agent_knowledge_documents(agent_id=agent_id)
    documents = db.list_agent_knowledge_documents(
        agent_id=agent_id,
        limit=pagination.page_size,
        offset=pagination.offset,
    )
    return KnowledgeDocumentSummaryPageResponse.create(
        items=[KnowledgeDocumentSummaryResponse(**document) for document in documents],
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.get(
    "/knowledge/documents",
    response_model=KnowledgeDocumentSummaryPageResponse,
    summary="查询全部知识库文档列表",
)
async def list_all_knowledge_documents(
    pagination: PaginationQuery = Depends(),
    db: Database = Depends(get_db),
    store: AgentStore = Depends(get_agent_store),
):
    return _build_knowledge_document_page(
        agent_id=None,
        pagination=pagination,
        store=store,
        db=db,
    )


@router.get(
    "/knowledge/documents/{agentId}",
    response_model=KnowledgeDocumentSummaryPageResponse,
    summary="查询 Agent 知识库文档标题与来源列表",
)
async def list_agent_knowledge_documents(
    agent_id: Annotated[str, Path(alias="agentId")],
    pagination: PaginationQuery = Depends(),
    db: Database = Depends(get_db),
    store: AgentStore = Depends(get_agent_store),
):
    return _build_knowledge_document_page(
        agent_id=agent_id,
        pagination=pagination,
        store=store,
        db=db,
    )


@router.get(
    "/{agentId}/knowledge/documents/{documentId}",
    response_model=KnowledgeDocumentDetailResponse,
    summary="按 document_id 查询 Agent 知识库原文",
)
async def get_agent_knowledge_document(
    agent_id: Annotated[str, Path(alias="agentId")],
    document_id: Annotated[str, Path(alias="documentId")],
    db: Database = Depends(get_db),
):
    document = db.get_agent_knowledge_document(agent_id=agent_id, document_id=document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"知识库文档不存在: {document_id}")
    return KnowledgeDocumentDetailResponse(**document)


@router.post(
    "/{agentId}/knowledge/search",
    response_model=KnowledgeSearchResponse,
    summary="仅检索当前 Agent 的知识库",
)
async def search_agent_knowledge(
    req: KnowledgeSearchRequest,
    agent_id: Annotated[str, Path(alias="agentId")],
    store: AgentStore = Depends(get_agent_store),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
):
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")

    try:
        sources = knowledge_base.search(
            collection_name=agent["collection_name"],
            query=req.query,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.error("[Agent:%s] search failed: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return KnowledgeSearchResponse(
        agent_id=agent_id,
        collection_name=agent["collection_name"],
        query=req.query,
        sources=_to_source_responses(sources),
    )


@router.post(
    "/{agentId}/chat",
    response_model=AgentChatResponse,
    summary="使用指定 Agent 对话（阻塞）",
)
async def agent_chat(
    req: AgentChatRequest,
    agent_id: Annotated[str, Path(alias="agentId")],
    store: AgentStore = Depends(get_agent_store),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    if not agent["is_active"]:
        raise HTTPException(status_code=403, detail="Agent 已停用")

    try:
        runner = AgentRunner(agent=agent, db=db, settings=settings)
        reply, session_id, sources = runner.chat(
            user_message=req.message,
            session_id=req.session_id,
        )
    except Exception as exc:
        logger.error("[Agent:%s] chat failed: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return AgentChatResponse(
        reply=reply,
        session_id=session_id,
        agent_id=agent_id,
        agent_name=agent["name"],
        model=agent["model"],
        provider=agent["provider"],
        sources=_to_source_responses(sources),
    )


@router.post(
    "/{agentId}/chat/stream",
    summary="使用指定 Agent 对话（SSE 流式）",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE 流式返回，done 事件中附带来源列表",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type":"chunk","content":"你好"}\n\n'
                        'data: {"type":"done","sessionId":"session-1","agentName":"法务助手","sources":[]}\n\n'
                    )
                }
            },
        }
    },
)
async def agent_chat_stream(
    req: AgentChatRequest,
    agent_id: Annotated[str, Path(alias="agentId")],
    store: AgentStore = Depends(get_agent_store),
    db: Database = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    agent = store.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent 不存在: {agent_id}")
    if not agent["is_active"]:
        raise HTTPException(status_code=403, detail="Agent 已停用")

    runner = AgentRunner(agent=agent, db=db, settings=settings)
    gen, session_id, sources = runner.chat_stream(
        user_message=req.message,
        session_id=req.session_id,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _run_stream(gen):
                yield _sse({"type": "chunk", "content": chunk})
            yield _sse(
                {
                    "type": "done",
                    "sessionId": session_id,
                    "agentId": agent_id,
                    "agentName": agent["name"],
                    "model": agent["model"],
                    "provider": agent["provider"],
                    "sources": [source.model_dump(by_alias=True) for source in _to_source_responses(sources)],
                }
            )
        except Exception as exc:
            logger.error("[Agent:%s] stream failed: %s", agent_id, exc)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
