"""会话管理 API — 列表、创建、历史、删除"""

import os
import json
import time
import uuid
import logging
from fastapi import APIRouter, Query

import config_data as config
from schemas.session import (
    SessionInfo, SessionListResponse, SessionCreateResponse,
    HistoryMessage, HistoryResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# 默认分页参数
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


def _extract_content(msg: dict) -> str:
    """从 LangChain message_to_dict 格式的消息中提取文本内容。
    兼容两种格式：嵌套 data.content 和顶层 content。"""
    if "data" in msg and isinstance(msg["data"], dict):
        return msg["data"].get("content", "")
    return msg.get("content", "")


def _get_history_path():
    """获取历史存储路径"""
    return config.chat_history_path


def _get_preview(session_id: str) -> str | None:
    """获取会话预览（第一条用户消息）"""
    file_path = os.path.join(_get_history_path(), f"{session_id}.json")
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
            for msg in messages:
                if msg.get("type") == "human":
                    content = _extract_content(msg)
                    return content[:50] + ("..." if len(content) > 50 else "")
            return "(空对话)"
    except Exception:
        logger.warning(f"读取会话预览失败: {session_id}", exc_info=True)
        return None


@router.get("", response_model=SessionListResponse)
def list_sessions(
    page: int = Query(default=1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="每页数量"),
):
    """列出所有会话（分页）"""
    history_path = _get_history_path()
    all_sessions = []

    if os.path.exists(history_path):
        for filename in os.listdir(history_path):
            if filename.endswith(".json") and "_summary" not in filename:
                session_id = filename.replace(".json", "")
                file_path = os.path.join(history_path, filename)
                mtime = os.path.getmtime(file_path)
                updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                preview = _get_preview(session_id)
                all_sessions.append(SessionInfo(
                    id=session_id,
                    preview=preview,
                    updated_at=updated_at
                ))

    # 按更新时间倒序
    all_sessions.sort(key=lambda s: s.updated_at, reverse=True)

    # 分页
    total = len(all_sessions)
    start = (page - 1) * page_size
    end = start + page_size
    paged = all_sessions[start:end]

    return SessionListResponse(
        sessions=paged,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=SessionCreateResponse)
def create_session():
    """创建新会话"""
    session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    history_path = _get_history_path()
    os.makedirs(history_path, exist_ok=True)
    file_path = os.path.join(history_path, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    return SessionCreateResponse(session_id=session_id)


@router.get("/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str):
    """获取会话的完整历史消息"""
    file_path = os.path.join(_get_history_path(), f"{session_id}.json")
    summary_path = os.path.join(_get_history_path(), f"{session_id}_summary.json")

    messages = []
    summary = None

    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_messages = json.load(f)
            for msg in raw_messages:
                role = "user" if msg.get("type") == "human" else "assistant"
                content = _extract_content(msg)
                messages.append(HistoryMessage(role=role, content=content))
    except Exception:
        logger.warning(f"读取会话历史失败: {session_id}", exc_info=True)

    try:
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
                summary = summary_data.get("summary", "")
    except Exception:
        logger.warning(f"读取会话摘要失败: {session_id}", exc_info=True)

    return HistoryResponse(messages=messages, summary=summary)


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """删除会话及其历史"""
    history_path = _get_history_path()
    file_path = os.path.join(history_path, f"{session_id}.json")
    summary_path = os.path.join(history_path, f"{session_id}_summary.json")

    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(summary_path):
        os.remove(summary_path)

    return {"status": "deleted", "session_id": session_id}
