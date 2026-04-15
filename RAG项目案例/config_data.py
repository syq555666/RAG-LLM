import os

md5_path = "./md5.text"
simhash_path = "./simhash_index.json"


#Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


#spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n","?","!","."," ","？","！","。",""]
max_split_char_number = 1000 #文本分割的阈值


# 检索参数
top_k = 3  # 检索返回的文档数量
score_threshold = 0.7  # 向量相似度阈值（0-1之间，越高越严格）


# 默认模型配置（可通过环境变量覆盖）
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen3-max")


# 会话配置
session_config = {
    "configurable": {
        "session_id": "default_session",
    }
}