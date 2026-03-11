from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

prompt = PromptTemplate.from_template("你是一个AI助手")
model = Tongyi(model="qwen3-max")

chain = prompt | model
#chain.invoke()
#chain.stream()

print(type(chain))