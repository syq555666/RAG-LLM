from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv
import os

#Chroma 向量数据库
#确保langchain-chroma chromadb 安装


load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

vector_store = Chroma(
    collection_name = "test",
    embedding_function = DashScopeEmbeddings(),
    persist_directory = "./chroma_db"
)

loader = CSVLoader(
    file_path = "./data/info.csv",
    source_column="source"
)


documents = loader.load()

#内存向量的新增
vector_store.add_documents(
    documents = documents,
    ids = ["id" + str(i) for i in range(1, len(documents)+1)]
)

#删除
vector_store.delete(["id1", "id2"])

#检索
result = vector_store.similarity_search(
    "中国最厉害的五所大学",
    5
)

print(result)