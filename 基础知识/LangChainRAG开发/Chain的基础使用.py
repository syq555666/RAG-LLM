from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")


chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human","请再来一首唐诗"),
    ]
)

history_data =[
    ("human","你来写一首诗"),
    ("ai","窗前明月光，疑是地上霜，举头望明月，低头思故乡。"),
    ("human","好诗，再来一个"),
    ("ai","锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
]

model = ChatTongyi(model="qwen3-max")

chain = chat_prompt_template | model

res = chain.invoke({"history": history_data})

print(res.content)

