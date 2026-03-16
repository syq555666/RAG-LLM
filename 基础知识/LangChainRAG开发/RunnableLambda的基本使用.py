from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

str_parser = StrOutputParser()

model = ChatTongyi(model="qwen3-max")

#第一个提示词模版
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname},刚生了{gender}，请帮忙起名字,仅告诉我名字，不要额外信息。"
)

#第二个提示词模版
second_prompt = PromptTemplate.from_template(
    "姓名：{name},请帮我解析含义"
)

my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

chain = first_prompt | model | my_func | second_prompt | model | str_parser

for chunk in chain.stream({"lastname": "曹" , "gender": "女儿"}):
    print(chunk, end="", flush=True )