"""FastAPI 依赖注入 — 管理服务单例"""

from services.knowledge_base_service import KnowledgeBaseService
from services.agent_service import AgentService
from services.history_store import SummarizingChatMessageHistory

# 模块级单例（由 lifespan 初始化）
_kb_service: KnowledgeBaseService | None = None
_agent_service: AgentService | None = None


def init_services():
    """在应用启动时初始化所有服务"""
    global _kb_service, _agent_service

    _kb_service = KnowledgeBaseService()
    _agent_service = AgentService(vector_store=_kb_service.chroma)

    return _kb_service, _agent_service


def get_kb_service() -> KnowledgeBaseService:
    """获取知识库服务单例"""
    assert _kb_service is not None, "KnowledgeBaseService 未初始化"
    return _kb_service


def get_agent_service() -> AgentService:
    """获取 Agent 服务单例"""
    assert _agent_service is not None, "AgentService 未初始化"
    return _agent_service


def get_history_store(session_id: str) -> SummarizingChatMessageHistory:
    """为指定会话创建历史存储实例"""
    return SummarizingChatMessageHistory(session_id)
