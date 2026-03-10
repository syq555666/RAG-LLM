from langchain_community.llms.tongyi import Tongyi
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

#不用qwen3-max，因为qwen3-max是聊天模型，qwen-max是大语言模型
model = Tongyi(model="qwen-max")

res = model.stream(input="你是谁，你能做什么？")

for chunk in res:
    print(chunk, end="", flush=True)