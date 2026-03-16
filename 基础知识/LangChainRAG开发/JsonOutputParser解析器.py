from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen3-max")

#第一个提示词模版
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname},刚生了{gender}，请帮忙起名字,"
    "并封装为json格式返回给我，要求key是name,value就是你起的名字，严格遵守这个格式"
)

#第二个提示词模版
second_prompt = PromptTemplate.from_template(
    "姓名：{name},请帮我解析含义"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser

res = chain.invoke({"lastname":"张", "gender":"女儿"})


#直接输出
#print(res)

#流式输出
for chunk in chain.stream({"lastname":"张", "gender":"女儿"}):
    print(chunk, end="", flush=True)