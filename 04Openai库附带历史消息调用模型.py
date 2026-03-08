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
        {"role": "system", "content": "你是一个AI助理，并且回答非常简洁"},
        {"role": "user", "content": "小明有两只宠物狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "小红有3只宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "一共有几只宠物"}
    ],
    stream=True   #流式输出
)

for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end=" ",
        flush=True
    )