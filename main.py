"""
main.py
FastAPI 应用入口。
启动命令: python main.py 或 uvicorn main:app --reload
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.agents import router as agents_router
from api.chat import router as chat_router
from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()


class RootResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    provider: str = Field(..., description="当前 LLM 提供方")
    model: str = Field(..., description="当前默认模型")
    docs: str = Field(..., description="Swagger 文档地址")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "running",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "docs": "/docs",
            }
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(..., description="健康检查结果")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
            }
        }
    }


app = FastAPI(
    title="Swagger 智能体平台 API",
    description="""

## 支持的 LLM Provider
- `openai` - OpenAI / 兼容接口 / Azure 等
- `bailian` - 阿里云百练（DashScope OpenAI 兼容模式）
- `anthropic` - Anthropic Claude
- `ollama` - 本地 Ollama 模型
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "agents", "description": "Agent 配置管理与对话"},
        {"name": "chat", "description": "基础对话接口"},
        {"name": "health", "description": "健康检查"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(agents_router)


@app.get("/", response_model=RootResponse, tags=["health"], summary="服务首页")
async def root():
    return RootResponse(
        status="running",
        provider=settings.llm_provider,
        model=settings.model_name,
        docs="/docs",
    )


@app.get("/health", response_model=HealthResponse, tags=["health"], summary="健康检查")
async def health():
    return HealthResponse(status="ok")


if __name__ == "__main__":
    logger.info("启动服务 http://%s:%s", settings.host, settings.port)
    logger.info("LLM Provider: %s | Model: %s", settings.llm_provider, settings.model_name)
    logger.info("Swagger 文档: http://localhost:%s/docs", settings.port)
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
