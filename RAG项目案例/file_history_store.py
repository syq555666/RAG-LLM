import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id        #会话id
        self.storage_path = storage_path    #不同会话id的存储文件的路径

        #完整的文件路径
        self.file_path = os.path.join(self.storage_path, f"{self.session_id}.json")

        #确保文件存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)


    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        #Sequence序列
        all_messages = list(self.messages)   #已有的消息列表
        all_messages.extend(messages)        #新的和已有的消息列表融合

        #将数据写入本地文件中（都是二进制）将BaseMessage消息转为字典
        new_messages = [message_to_dict(message) for message in all_messages]

        #将数据写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property    #@property装饰器可以将messages方法变为成员属性
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)   #list里都是字典
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)