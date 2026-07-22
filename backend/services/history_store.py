"""对话历史存储 — 文件持久化 + 自动摘要 + 并发安全"""

import json
import os
import threading
import logging
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config_data as config

logger = logging.getLogger(__name__)

HISTORY_SUMMARY_THRESHOLD = 10
RECENT_MESSAGE_KEEP_COUNT = 6

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """请将以下对话历史总结成简短的摘要，保留关键信息（如用户问题、AI回答要点）。

要求：
1. 摘要不超过100字
2. 只保留关键信息
3. 使用中文"""),
        ("user", "对话历史：\n{history}")
    ]
)

# 文件级锁 — 防止同一会话并发读写导致 JSON 损坏
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _acquire_file_lock(file_path: str) -> threading.Lock:
    """获取指定文件的锁（线程安全地创建/获取）"""
    with _file_locks_guard:
        if file_path not in _file_locks:
            _file_locks[file_path] = threading.Lock()
        return _file_locks[file_path]


class SummarizingChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path=None):
        self.session_id = session_id
        if storage_path is None:
            storage_path = config.chat_history_path
        self.storage_path = storage_path
        self.summary_file_path = os.path.join(storage_path, f"{session_id}_summary.json")
        self.file_path = os.path.join(storage_path, f"{session_id}.json")
        os.makedirs(storage_path, exist_ok=True)

        # 获取此会话文件的锁
        self._lock = _acquire_file_lock(self.file_path)
        self._summary_chain = None

    def _get_summary_chain(self):
        """懒加载总结 chain（使用共享 LLM）"""
        if self._summary_chain is None:
            from api.deps import get_shared_llm
            self._summary_chain = SUMMARY_PROMPT | get_shared_llm() | StrOutputParser()
        return self._summary_chain

    def _get_summary(self) -> str:
        """获取历史摘要"""
        try:
            if os.path.exists(self.summary_file_path):
                with open(self.summary_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("summary", "")
        except Exception:
            logger.warning(f"读取会话摘要失败: {self.session_id}", exc_info=True)
        return ""

    def _save_summary(self, summary: str):
        """保存历史摘要"""
        with open(self.summary_file_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False)

    def _summarize_and_truncate(self, messages: list):
        """总结历史并截断，只保留摘要+最近消息。"""
        if len(messages) < HISTORY_SUMMARY_THRESHOLD:
            return

        existing_summary = self._get_summary()

        history_text = ""
        if existing_summary:
            history_text += f"[之前的对话摘要]\n{existing_summary}\n\n[最近对话]\n"

        for msg in messages:
            role = "用户" if msg.type == "human" else "AI"
            history_text += f"{role}: {msg.content}\n"

        try:
            summary = self._get_summary_chain().invoke({"history": history_text}).strip()
            self._save_summary(summary)

            recent_count = min(RECENT_MESSAGE_KEEP_COUNT, len(messages))
            recent_messages = messages[-recent_count:]
            new_messages = [message_to_dict(message) for message in recent_messages]

            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(new_messages, f, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"总结历史失败 (session={self.session_id}): {e}")

    @property
    def messages(self) -> list[BaseMessage]:
        with self._lock:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    messages_data = json.load(f)
                    return messages_from_dict(messages_data)
            except FileNotFoundError:
                return []
            except Exception:
                logger.warning(f"读取会话消息失败: {self.session_id}", exc_info=True)
                return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        with self._lock:
            all_messages = list(self.messages)
            all_messages.extend(messages)

            new_messages = [message_to_dict(message) for message in all_messages]

            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(new_messages, f, ensure_ascii=False)

            if len(all_messages) >= HISTORY_SUMMARY_THRESHOLD:
                self._summarize_and_truncate(all_messages)

    def get_context_for_llm(self) -> str:
        """获取传递给 LLM 的上下文（包括摘要 + 最近消息）"""
        with self._lock:
            summary = self._get_summary()
            recent_messages = self.messages[-RECENT_MESSAGE_KEEP_COUNT:]

        context = ""
        if summary:
            context += f"[之前对话摘要]\n{summary}\n\n"

        for msg in recent_messages:
            role = "用户" if msg.type == "human" else "AI"
            context += f"{role}: {msg.content}\n"

        return context

    def clear(self) -> None:
        with self._lock:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            if os.path.exists(self.summary_file_path):
                os.remove(self.summary_file_path)
