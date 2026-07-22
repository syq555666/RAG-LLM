"""FastAPI 依赖注入 — 管理服务单例"""

import threading
from services.knowledge_base_service import KnowledgeBaseService
from services.agent_service import AgentService
from services.history_store import SummarizingChatMessageHistory
from langchain_deepseek import ChatDeepSeek
import config_data as config

# 模块级单例（由 lifespan 初始化）
_kb_service: KnowledgeBaseService | None = None
_agent_service: AgentService | None = None
_shared_llm: ChatDeepSeek | None = None


def _get_shared_llm() -> ChatDeepSeek:
    """获取共享的 LLM 实例（懒加载单例）"""
    global _shared_llm
    if _shared_llm is None:
        _shared_llm = ChatDeepSeek(model=config.chat_model_name)
    return _shared_llm


def init_services():
    """在应用启动时初始化所有服务"""
    global _kb_service, _agent_service

    _kb_service = KnowledgeBaseService()
    _agent_service = AgentService(
        vector_store=_kb_service.chroma,
        chat_model=_get_shared_llm(),
    )

    # 注册知识库变更回调：当知识库内容变化时，通知 Agent 重建检索索引
    _kb_service.on_change = lambda: _agent_service.invalidate_retriever()

    return _kb_service, _agent_service


def get_kb_service() -> KnowledgeBaseService:
    """获取知识库服务单例"""
    if _kb_service is None:
        raise RuntimeError("KnowledgeBaseService 未初始化，请先调用 init_services()")
    return _kb_service


def get_agent_service() -> AgentService:
    """获取 Agent 服务单例"""
    if _agent_service is None:
        raise RuntimeError("AgentService 未初始化，请先调用 init_services()")
    return _agent_service


def get_shared_llm() -> ChatDeepSeek:
    """获取共享的 ChatDeepSeek 实例（供外部模块使用）"""
    return _get_shared_llm()


def get_history_store(session_id: str) -> SummarizingChatMessageHistory:
    """为指定会话创建历史存储实例"""
    return SummarizingChatMessageHistory(session_id)
