import os

# 基础目录 - backend/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 去重索引文件路径
md5_path = os.path.join(DATA_DIR, "md5.json")
simhash_path = os.path.join(DATA_DIR, "simhash_index.json")

# Chroma 向量数据库
collection_name = "rag"
persist_directory = os.path.join(DATA_DIR, "chroma_db")

# 对话历史存储路径
chat_history_path = os.path.join(DATA_DIR, "chat_history")

# 文本分割器配置
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", "?", "!", ".", " ", "？", "！", "。", ""]
max_split_char_number = 1000  # 文本分割的阈值

# 检索参数
top_k = 3  # 检索返回的文档数量

# 默认模型配置（可通过环境变量覆盖）
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen3-max")
