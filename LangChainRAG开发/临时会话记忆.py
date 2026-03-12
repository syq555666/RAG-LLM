from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

model = ChatTongyi(model = "qwen3-max")
prompt = PromptTemplate.from_template(
    "你需要根据会话历史回应用户问题。对话历史: {chat_history}, 用户提问: {input}，请回答"
)

str_parser = StrOutputParser()

base_chain = prompt | model | str_parser

#创建一个新的链，对原有的链增强功能：自动附加历史消息

store = {}   #key就是session_id, value是InMemoryChatMessageHistory

def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

conversation_chain = RunnableWithMessageHistory(
    base_chain,   #原有的链条
    get_history,  #通过会话获取历史消息
    input_messages_key = "input",
    history_messages_key = "chat_history"
)

if __name__ == "__main__":
    #添加langchain配置
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }
    res = conversation_chain.invoke({"input": "小明有两只猫"},session_config)
    print("第1次执行：", res)

    res = conversation_chain.invoke({"input": "小明有三只狗"},session_config)
    print("第2次执行：", res)

    res = conversation_chain.invoke({"input": "小明有几只宠物"}, session_config)
    print("第3次执行：", res)