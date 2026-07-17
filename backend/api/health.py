from fastapi import APIRouter, Depends
from api.deps import get_kb_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check():
    """存活检查 + 知识库状态"""
    try:
        kb = get_kb_service()
        kb_count = len(kb.get_file_list())
    except Exception:
        kb_count = 0

    return {"status": "ok", "kb_file_count": kb_count}
