"""健康检查 API"""

import logging
from fastapi import APIRouter
from api.deps import get_kb_service, get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check():
    """存活检查 + 知识库状态
    get_file_list() 现在从内存缓存读取，O(1) 无全量 Chroma 查询。
    """
    try:
        kb = get_kb_service()
        kb_count = len(kb.get_file_list())
        agent_cache_size = len(get_agent_service()._cache)
        return {
            "status": "ok",
            "kb_file_count": kb_count,
            "agent_cache_entries": agent_cache_size,
        }
    except RuntimeError as e:
        return {"status": "error", "kb_file_count": 0, "detail": str(e)}
    except Exception as e:
        logger.warning(f"健康检查降级: {e}")
        return {"status": "degraded", "kb_file_count": 0, "detail": str(e)}
