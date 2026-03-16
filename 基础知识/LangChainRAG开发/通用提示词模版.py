from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")


prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}，刚生了{gender}，你帮我起个名字，简单回答。"
)

prompt_text = prompt_template.format(lastname="张",gender="女儿")

model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res)

chain = prompt_template | model

res = chain.invoke(input={"lastname": "张","gender": "女儿"})
print(res)