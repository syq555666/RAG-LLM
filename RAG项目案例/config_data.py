import os

md5_path = "./md5.text"
simhash_path = "./simhash_index.json"


#Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


# 临时修复：使用内存模式（如果持久化失败）
# persist_directory = ":memory:"

#spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n","?","!","."," ","？","！","。",""]
max_split_char_number = 1000 #文本分割的阈值


# 检索参数
top_k = 5  # 检索返回的文档数量
score_threshold = 0.5  # 向量相似度阈值（0-1之间，越高越严格）


embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"


session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }