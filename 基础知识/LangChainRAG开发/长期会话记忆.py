from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from typing import Sequence
import os, json
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

model = ChatTongyi(model="qwen3-max")

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



prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "你需要根据会话历史回应用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
    ]
)

str_parser = StrOutputParser()

base_chain = prompt | model | str_parser

def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")

# 创建一个新的链，对原有的链增强功能：自动附加历史消息
conversation_chain = RunnableWithMessageHistory(
    base_chain,  # 原有的链条
    get_history,  # 通过会话获取历史消息
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    # 添加langchain配置
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    #res = conversation_chain.invoke({"input": "小明有两只猫"}, session_config)
    #print("第1次执行：", res)

    #res = conversation_chain.invoke({"input": "小明有三只狗"}, session_config)
    #print("第2次执行：", res)

    res = conversation_chain.invoke({"input": "小明有几只宠物"}, session_config)
    print("第3次执行：", res)