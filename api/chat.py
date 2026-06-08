"""
api/chat.py
基础对话接口，支持阻塞响应与 SSE 流式输出。
"""

import asyncio
import json
import logging
import uuid
from typing import Annotated, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse
from pydantic import Field

from api.camel_model import CamelModel
from config.settings import Settings, get_settings
from core.llm_client import LLMClient
from storage.db import Database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_db: Optional[Database] = None
_llm: Optional[LLMClient] = None


def get_db(settings: Settings = Depends(get_settings)) -> Database:
    global _db
    if _db is None:
        _db = Database(settings.db_path)
    return _db


def get_llm(settings: Settings = Depends(get_settings)) -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient(settings)
    return _llm


class ChatRequest(CamelModel):
    message: str = Field(..., description="用户输入", min_length=1)
    system_prompt: str = Field(
        default="你是一个聪明、专业的 AI 助手，请用简洁清晰的中文回答问题。",
        description="系统提示词",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class SessionChatRequest(ChatRequest):
    session_id: Optional[str] = Field(default=None, description="会话 ID，为空时自动创建")


class ChatResponse(CamelModel):
    reply: str = Field(..., description="模型回复")
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    provider: str = Field(..., description="模型提供方")
    model: str = Field(..., description="模型名称")

    model_config = CamelModel.model_config | {
        "json_schema_extra": {
            "example": {
                "reply": "你好，我可以帮你分析这段代码。",
                "sessionId": "3bc81d84-66de-4d4b-a040-52b2ea0a9001",
                "provider": "openai",
                "model": "gpt-4o-mini",
            }
        }
    }


class HistoryMessageResponse(CamelModel):
    role: str = Field(..., description="消息角色，如 user / assistant")
    content: str = Field(..., description="消息内容")
    created_at: str = Field(..., description="消息创建时间")

    model_config = CamelModel.model_config | {
        "json_schema_extra": {
            "example": {
                "role": "user",
                "content": "请解释一下这段代码。",
                "createdAt": "2026-05-26T10:00:00.000000",
            }
        }
    }


class HistoryResponse(CamelModel):
    session_id: str = Field(..., description="会话 ID")
    messages: list[HistoryMessageResponse] = Field(..., description="会话消息列表")

    model_config = CamelModel.model_config | {
        "json_schema_extra": {
            "example": {
                "sessionId": "3bc81d84-66de-4d4b-a040-52b2ea0a9001",
                "messages": [
                    {
                        "role": "user",
                        "content": "请解释一下这段代码。",
                        "createdAt": "2026-05-26T10:00:00.000000",
                    },
                    {
                        "role": "assistant",
                        "content": "这段代码的作用是读取配置并初始化客户端。",
                        "createdAt": "2026-05-26T10:00:05.000000",
                    },
                ],
            }
        }
    }


class SessionSummaryResponse(CamelModel):
    id: str = Field(..., description="会话 ID")
    created_at: str = Field(..., description="会话创建时间")
    title: str = Field(..., description="会话标题")
    last_msg: Optional[str] = Field(default=None, description="最后一条消息")

    model_config = CamelModel.model_config | {
        "json_schema_extra": {
            "example": {
                "id": "3bc81d84-66de-4d4b-a040-52b2ea0a9001",
                "createdAt": "2026-05-26T10:00:00.000000",
                "title": "请解释一下这段代码",
                "lastMsg": "这段代码的作用是读取配置并初始化客户端。",
            }
        }
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_stream(generator) -> AsyncGenerator[str, None]:
    loop = asyncio.get_event_loop()
    sentinel = object()

    def next_chunk(it):
        try:
            return next(it)
        except StopIteration:
            return sentinel

    it = iter(generator)
    while True:
        chunk = await loop.run_in_executor(None, next_chunk, it)
        if chunk is sentinel:
            break
        yield chunk


@router.post("", response_model=ChatResponse, summary="单次对话（阻塞）")
async def chat_once(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm),
):
    try:
        reply = llm.chat(
            messages=[{"role": "user", "content": req.message}],
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return ChatResponse(
            reply=reply,
            provider=settings.llm_provider,
            model=settings.model_name,
        )
    except Exception as exc:
        logger.error("[/chat] failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/session", response_model=ChatResponse, summary="多轮对话（阻塞）")
async def chat_with_session(
    req: SessionChatRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm),
    db: Database = Depends(get_db),
):
    session_id = req.session_id or str(uuid.uuid4())
    db.create_session(session_id, title=req.message[:30])
    history = db.get_history_for_llm(session_id)
    history.append({"role": "user", "content": req.message})

    try:
        reply = llm.chat(
            messages=history,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as exc:
        logger.error("[/chat/session] failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    db.add_message(session_id, "user", req.message)
    db.add_message(session_id, "assistant", reply)
    return ChatResponse(
        reply=reply,
        session_id=session_id,
        provider=settings.llm_provider,
        model=settings.model_name,
    )


@router.post(
    "/stream",
    summary="单次对话 SSE 流式",
    description="逐字返回，前端可用 EventSource 或 fetch + ReadableStream 接收。",
    tags=["chat"],
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE 事件流",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type":"chunk","content":"你好"}\n\n'
                        'data: {"type":"done","model":"gpt-4o-mini","provider":"openai"}\n\n'
                    )
                }
            },
        }
    },
)
async def chat_stream(
    req: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm),
):
    async def event_generator() -> AsyncGenerator[str, None]:
        full_reply: list[str] = []
        try:
            gen = llm.chat_stream(
                messages=[{"role": "user", "content": req.message}],
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            async for chunk in _run_stream(gen):
                full_reply.append(chunk)
                yield _sse({"type": "chunk", "content": chunk})

            yield _sse(
                {
                    "type": "done",
                    "model": settings.model_name,
                    "provider": settings.llm_provider,
                }
            )
            logger.info("[/chat/stream] completed, total chars=%s", len("".join(full_reply)))
        except Exception as exc:
            logger.error("[/chat/stream] error: %s", exc)
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


@router.post(
    "/session/stream",
    summary="多轮对话 SSE 流式（带会话记忆）",
    description="""
带 SQLite 会话记忆的流式接口：
- 不传 `sessionId` 时自动创建新会话
- 流结束后自动将本轮消息持久化到数据库
""",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "带会话 ID 的 SSE 事件流",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"type":"chunk","content":"你好"}\n\n'
                        'data: {"type":"done","sessionId":"session-1","model":"gpt-4o-mini","provider":"openai"}\n\n'
                    )
                }
            },
        }
    },
)
async def chat_session_stream(
    req: SessionChatRequest,
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm),
    db: Database = Depends(get_db),
):
    session_id = req.session_id or str(uuid.uuid4())
    db.create_session(session_id, title=req.message[:30])
    history = db.get_history_for_llm(session_id)
    history.append({"role": "user", "content": req.message})

    async def event_generator() -> AsyncGenerator[str, None]:
        full_reply: list[str] = []
        try:
            gen = llm.chat_stream(
                messages=history,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            async for chunk in _run_stream(gen):
                full_reply.append(chunk)
                yield _sse({"type": "chunk", "content": chunk})

            complete_reply = "".join(full_reply)
            db.add_message(session_id, "user", req.message)
            db.add_message(session_id, "assistant", complete_reply)

            yield _sse(
                {
                    "type": "done",
                    "sessionId": session_id,
                    "model": settings.model_name,
                    "provider": settings.llm_provider,
                }
            )
            logger.info("[/chat/session/stream] session=%s completed, total chars=%s", session_id, len(complete_reply))
        except Exception as exc:
            logger.error("[/chat/session/stream] error: %s", exc)
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


@router.get("/history/{sessionId}", response_model=HistoryResponse, summary="查询会话历史")
async def get_history(session_id: Annotated[str, Path(alias="sessionId")], db: Database = Depends(get_db)):
    return HistoryResponse(session_id=session_id, messages=db.get_messages(session_id))


@router.get(
    "/sessions",
    response_model=list[SessionSummaryResponse],
    summary="查询最近会话列表",
)
async def list_sessions(limit: int = 20, db: Database = Depends(get_db)):
    return [SessionSummaryResponse(**session) for session in db.list_sessions(limit=limit)]
