import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import config_data as config
from dotenv import load_dotenv

load_dotenv()

# 历史消息数量阈值，超过则触发总结
HISTORY_SUMMARY_THRESHOLD = 10

# 总结历史的 prompt
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


class SummarizingChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.summary_file_path = os.path.join(storage_path, f"{session_id}_summary.json")
        self.file_path = os.path.join(storage_path, f"{session_id}.json")
        os.makedirs(storage_path, exist_ok=True)

        # 初始化 LLM 用于总结
        self.llm = ChatTongyi(model=config.chat_model_name)
        self.summary_chain = SUMMARY_PROMPT | self.llm | StrOutputParser()

    def _get_summary(self) -> str:
        """获取历史摘要"""
        try:
            if os.path.exists(self.summary_file_path):
                with open(self.summary_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("summary", "")
        except Exception:
            pass
        return ""

    def _save_summary(self, summary: str):
        """保存历史摘要"""
        with open(self.summary_file_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary}, f, ensure_ascii=False)

    def _summarize_history(self, messages: list):
        """总结历史消息"""
        if len(messages) < HISTORY_SUMMARY_THRESHOLD:
            return

        # 将消息转为文本格式
        history_text = ""
        for msg in messages:
            role = "用户" if msg.type == "human" else "AI"
            history_text += f"{role}: {msg.content}\n"

        try:
            summary = self.summary_chain.invoke({"history": history_text}).strip()
            self._save_summary(summary)
        except Exception as e:
            print(f"总结历史失败: {e}")

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages = [message_to_dict(message) for message in all_messages]

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f, ensure_ascii=False)

        # 检查是否需要总结历史
        if len(all_messages) >= HISTORY_SUMMARY_THRESHOLD:
            self._summarize_history(all_messages)

    def get_context_for_llm(self) -> str:
        """获取传递给 LLM 的上下文（包括摘要 + 最近消息）"""
        summary = self._get_summary()
        recent_messages = self.messages[-6:]  # 只保留最近6条

        context = ""
        if summary:
            context += f"[之前对话摘要]\n{summary}\n\n"

        for msg in recent_messages:
            role = "用户" if msg.type == "human" else "AI"
            context += f"{role}: {msg.content}\n"

        return context

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        if os.path.exists(self.summary_file_path):
            os.remove(self.summary_file_path)


def get_history(session_id):
    return SummarizingChatMessageHistory(session_id, "./chat_history")