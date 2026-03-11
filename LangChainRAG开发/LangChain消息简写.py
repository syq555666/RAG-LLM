from langchain_community.chat_models.tongyi import ChatTongyi
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

#得到模型对象
model = ChatTongyi(model="qwen3-max")

#准备消息列表
messages = [
    #角色：system/human/ai
    ("system", "你是一个边塞诗人。"),
    ("human", "写一首唐诗。"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "按照上一个回复的格式，再写一首唐诗。")
]

#流失输出
res = model.stream(input=messages)

#通过.content获取内容
for chunk in res:
    print(chunk.content, end="", flush=True)