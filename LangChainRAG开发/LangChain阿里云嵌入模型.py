from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 从环境变量获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")


#创建模型对象 不传model默认用的是text-embeddings-v1
model = DashScopeEmbeddings()


print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你", "晚上吃啥"]))