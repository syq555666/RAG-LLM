from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import os
from dotenv import load_dotenv

parser = StrOutputParser()

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

model =ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我领居姓：{lastname}, 刚生了{gender},请起名，仅告诉我名字无需其他内容。"
)

chain = prompt | model | parser | model

res = chain.invoke({"lastname":"张","gender":"女儿"})

print(res.content)