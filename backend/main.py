import sys
import os

# 确保 backend 目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.deps import init_services
from api.health import router as health_router
from api.knowledge import router as knowledge_router
from api.chat import router as chat_router
from api.sessions import router as sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 — 启动时初始化服务单例"""
    print("🚀 正在初始化服务...")
    kb_service, agent_service = init_services()
    print(f"✅ 知识库服务已启动，当前 {len(kb_service.get_file_list())} 个文件")
    print("✅ Agent 服务已就绪")
    yield
    print("👋 应用已关闭")


app = FastAPI(
    title="RAG-LLM 智能客服 API",
    description="基于 LangChain + Chroma 的 RAG 智能客服系统后端",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 配置
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health_router)
app.include_router(knowledge_router)
app.include_router(chat_router)
app.include_router(sessions_router)


@app.get("/")
def root():
    """根路径 — API 信息"""
    return {
        "name": "RAG-LLM 智能客服 API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
