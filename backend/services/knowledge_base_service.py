# 知识库服务
import os
import hashlib
import json
import threading
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from simhash import Simhash

import config_data as config

# SimHash 相似度阈值 (0-1，越高越严格)
SIMHASH_SIMILARITY_THRESHOLD = 0.8


def _load_json(path: str, default=None):
    """安全加载 JSON 文件"""
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _save_json(path: str, data):
    """安全保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


# ---- MD5 去重（文件级，格式: {md5: [filename, ...]}） ----

def check_md5(md5_str: str) -> bool:
    """检查传入的 md5 字符串是否已经被处理过"""
    index = _load_json(config.md5_path, {})
    return md5_str in index


def save_md5(md5_str: str, filename: str):
    """将 md5 与文件名关联保存"""
    index = _load_json(config.md5_path, {})
    if md5_str not in index:
        index[md5_str] = []
    if filename not in index[md5_str]:
        index[md5_str].append(filename)
    _save_json(config.md5_path, index)


def remove_md5_for_file(filename: str):
    """从 MD5 索引中移除指定文件的所有记录"""
    index = _load_json(config.md5_path, {})
    to_delete = [k for k, v in index.items() if filename in v]
    for k in to_delete:
        index[k].remove(filename)
        if not index[k]:
            del index[k]
    _save_json(config.md5_path, index)


def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转为 md5 字符串"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()


# ---- SimHash 近似去重（chunk 级，格式: [{"hash": int, "source": str}]） ----

def load_simhash_index() -> list:
    """加载 SimHash 索引（兼容旧格式）"""
    if not os.path.exists(config.simhash_path):
        return []
    try:
        with open(config.simhash_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    # 迁移旧格式：旧格式是纯整数列表 [123, 456, ...]
    if data and isinstance(data[0], (int, float)):
        migrated = [{"hash": int(h), "source": "unknown"} for h in data]
        _save_json(config.simhash_path, migrated)
        return migrated

    return data


def save_simhash_index(index: list):
    """保存 SimHash 索引"""
    _save_json(config.simhash_path, index)


def remove_simhash_for_file(filename: str):
    """从 SimHash 索引中移除指定文件的条目"""
    index = load_simhash_index()
    index = [entry for entry in index if entry.get("source") != filename]
    save_simhash_index(index)


def compute_similarity(h1: int, h2: int) -> float:
    """计算两个 SimHash 的相似度 (基于汉明距离)"""
    distance = (h1 ^ h2).bit_count()  # 汉明距离
    similarity = 1 - (distance / 64.0)  # 64位 SimHash
    return similarity


def is_similar(text: str, existing_hashes: list) -> bool:
    """检查文本是否与已有文本相似（existing_hashes 是预计算好的 hash 整数列表）"""
    if not existing_hashes:
        return False

    current_hash = Simhash(text).value
    for existing_val in existing_hashes:
        if compute_similarity(current_hash, existing_val) >= SIMHASH_SIMILARITY_THRESHOLD:
            return True
    return False


class KnowledgeBaseService:
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        # 使用阿里云的 Embedding
        from langchain_community.embeddings import DashScopeEmbeddings
        embedding = DashScopeEmbeddings(
            model=config.embedding_model_name
        )

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=embedding,
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

        # 线程安全锁 — Chroma 不支持并发写
        self._write_lock = threading.Lock()

    def _get_existing_hash_values(self, exclude_source: str = None) -> list:
        """获取当前索引中所有 hash 值（排除指定 source，用于去重检测）"""
        values = []
        for entry in self.simhash_index:
            if exclude_source and entry.get("source") == exclude_source:
                continue
            h = entry.get("hash")
            if h is not None:
                values.append(int(h))
        return values

    def upload_by_str(self, data, filename):
        """将传入的字符串进行向量化，存入向量数据库中（线程安全）"""
        with self._write_lock:
            return self._upload_by_str_unsafe(data, filename)

    def _upload_by_str_unsafe(self, data, filename):
        """内部实现：不安全的写入操作"""
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
        skipped_count = 0
        existing_hash_values = self._get_existing_hash_values()

        for chunk in knowledge_chunks:
            if is_similar(chunk, existing_hash_values):
                skipped_count += 1
            else:
                new_chunks.append(chunk)
                hash_val = Simhash(chunk).value
                self.simhash_index.append({"hash": hash_val, "source": filename})
                existing_hash_values.append(hash_val)  # 同一批次内也去重

        # 批量保存 SimHash 索引（一次磁盘写入）
        save_simhash_index(self.simhash_index)

        if not new_chunks:
            save_md5(md5_hex, filename)
            return f"【跳过】，{skipped_count} 个段落与已有内容相似"

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "system",
        }

        self.chroma.add_texts(
            new_chunks,
            metadatas=[metadata for _ in new_chunks],
        )

        save_md5(md5_hex, filename)

        skipped_info = f"，跳过了 {skipped_count} 个相似段落" if skipped_count else ""
        return f"【成功】，{len(new_chunks)} 个新段落已入库{skipped_info}"

    def delete_by_filename(self, filename: str):
        """根据文件名删除所有关联的 chunk（线程安全），并精准清理去重索引"""
        with self._write_lock:
            self.chroma.delete(where={"source": filename})

            # 精准清理：只移除该文件的记录
            remove_md5_for_file(filename)
            remove_simhash_for_file(filename)

            # 重新加载内存中的 simhash 索引
            self.simhash_index = load_simhash_index()

    def get_file_list(self) -> set:
        """获取知识库中的文件列表"""
        try:
            collection = self.chroma.get()
            metadatas = collection.get('metadatas', []) if collection else []
            return set(m.get('source') for m in metadatas if m and m.get('source'))
        except Exception:
            return set()
