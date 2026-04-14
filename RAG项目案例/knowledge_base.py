#知识库
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from dotenv import load_dotenv
from simhash import Simhash
import json

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

# SimHash 相似度阈值 (0-1，越高越严格)
SIMHASH_SIMILARITY_THRESHOLD = 0.8


def check_md5(md5_str: str):
    #检查传入的md5字符串是否已经被处理过了
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r',encoding='utf-8').readlines():
            line = line.strip()     #处理字符串前后的空格和回车
            if line ==md5_str:
                return True
        return False



def save_md5(md5_str : str):
    #将传入的md5字符串，记录到文件内保存
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    #将传入的字符串转为md5字符串
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


def load_simhash_index():
    """加载 SimHash 索引"""
    if not os.path.exists(config.simhash_path):
        return []
    try:
        with open(config.simhash_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_simhash_index(index):
    """保存 SimHash 索引"""
    with open(config.simhash_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False)


def compute_similarity(hash1, hash2):
    """计算两个 SimHash 的相似度 (基于汉明距离)"""
    # 获取 Simhash 对象的 value (整数)
    if hasattr(hash1, 'value'):
        h1 = hash1.value
    else:
        h1 = hash1
    if hasattr(hash2, 'value'):
        h2 = hash2.value
    else:
        h2 = hash2
    distance = (h1 ^ h2).bit_count()  # 汉明距离
    similarity = 1 - (distance / 64.0)       # 64位 SimHash
    return similarity


def is_similar(text: str, existing_hashes: list) -> bool:
    """检查文本是否与已有文本相似"""
    if not existing_hashes:
        return False

    current_hash = Simhash(text)
    for existing_hash in existing_hashes:
        # existing_hashes 存储的是整数
        existing_val = Simhash(existing_hash) if isinstance(existing_hash, str) else existing_hash
        if compute_similarity(current_hash, existing_val) >= SIMHASH_SIMILARITY_THRESHOLD:
            return True
    return False


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        # 加载已有的 SimHash 索引
        self.simhash_index = load_simhash_index()

    def upload_by_str(self, data, filename):
        # 将传入的字符串，进行向量化，存入向量数据库中
        # 先检查 MD5 (精确去重)
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "【跳过】，内容已经存在知识库中"

        # 文本分割
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 过滤相似内容 (SimHash 近似去重)
        new_chunks = []
        skipped_chunks = []
        existing_hashes = self.simhash_index.copy()

        for chunk in knowledge_chunks:
            if is_similar(chunk, existing_hashes):
                skipped_chunks.append(chunk)
            else:
                new_chunks.append(chunk)
                # 将新 chunk 的 SimHash 加入索引
                self.simhash_index.append(Simhash(chunk).value)

        if not new_chunks:
            # 所有 chunk 都是相似的，记录 MD5 但不新增
            save_md5(md5_hex)
            return f"【跳过】，{len(skipped_chunks)} 个段落与已有内容相似"

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小王",
        }

        self.chroma.add_texts(
            new_chunks,
            metadatas=[metadata for _ in new_chunks],
        )

        # 保存 SimHash 索引
        save_simhash_index(self.simhash_index)
        save_md5(md5_hex)

        # 返回结果包含跳过数量
        skipped_info = f"，跳过了 {len(skipped_chunks)} 个相似段落" if skipped_chunks else ""
        return f"【成功】，{len(new_chunks)} 个新段落已入库{skipped_info}"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    # 测试 SimHash 相似检测
    print("测试 SimHash 去重:")
    print(f"'羽绒服要手洗' vs '羽绒服要干洗': {compute_similarity(Simhash('羽绒服要手洗'), Simhash('羽绒服要干洗')):.2f}")
    print(f"'羽绒服要手洗' vs '羽绒服要手洗': {compute_similarity(Simhash('羽绒服要手洗'), Simhash('羽绒服要手洗')):.2f}")