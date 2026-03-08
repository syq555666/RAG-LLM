from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "你是一个编程Python专家，并且话非常多"},
        {"role": "assistant", "content": "好的，我是编程专家，并且话非常多，你要问什么"},
        {"role": "user", "content": "输出1-10的数字，使用python代码"}
    ],
    stream=True   #流式输出
)

for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end=" ",
        flush=True
    )