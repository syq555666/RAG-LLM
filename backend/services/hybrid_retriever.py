import logging
from langchain_core.documents import Document
import config_data as config
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """分词：中文用 jieba，其他用空格分割"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 回退：无 jieba 时按字符 bigram 分词（对中文比纯空格好得多）
        result = []
        for i in range(len(text) - 1):
            result.append(text[i:i+2])
        return result if result else text.split()


class HybridRetriever:
    """混合检索器：结合向量检索和BM25关键词检索
    自动检测 Chroma 文档数量变化，在知识库更新后重建 BM25 索引。
    """

    def __init__(self, vector_store, k: int = 5, bm25_threshold: float = 1.0):
        self.vector_store = vector_store
        self.k = k
        self.bm25 = None
        self.chunk_texts = []
        self._index_built = False
        self._indexed_doc_count = 0  # 记录构建索引时的文档数
        self.bm25_threshold = bm25_threshold

    def invalidate(self):
        """强制下次检索时重建 BM25 索引（由外部在知识库变更后调用）"""
        if self._index_built:
            self._index_built = False
            logger.info("BM25 索引已失效，下次检索时将重建")

    def _get_current_doc_count(self) -> int:
        """获取 Chroma 中当前文档数量（轻量操作，仅 COUNT 查询）"""
        try:
            # Chroma 内部 collection.count() 执行 SQL COUNT，O(1)
            return self.vector_store._collection.count()
        except Exception:
            return -1  # 异常时强制重建

    def _build_bm25_index(self):
        """从Chroma获取所有文档构建BM25索引"""
        all_data = self.vector_store.get()
        documents = all_data.get("documents", [])

        if documents:
            self.chunk_texts = documents
            tokenized_corpus = [_tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._index_built = True
            self._indexed_doc_count = len(documents)
            logger.info(f"BM25 索引已构建，文档数: {len(documents)}")
        else:
            self._index_built = True
            self._indexed_doc_count = 0
            logger.info("BM25 索引为空（知识库无文档）")

    def invoke(self, query: str) -> list[Document]:
        """执行混合检索，返回融合后的文档。
        每次调用前检查文档数是否变化，自动重建过期索引。
        """
        # 检查索引是否失效（文档数变化）
        if self._index_built:
            current_count = self._get_current_doc_count()
            if current_count >= 0 and current_count != self._indexed_doc_count:
                logger.info(f"检测到文档数变化 ({self._indexed_doc_count} → {current_count})，重建 BM25 索引")
                self._index_built = False

        if not self._index_built:
            self._build_bm25_index()

        # 向量检索
        vector_results = self.vector_store.similarity_search(query, k=self.k)

        # BM25检索（中文分词）
        bm25_docs: list[tuple[str, int | None]] = []
        if self._index_built and self.bm25:
            query_tokens = _tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(bm25_scores)[-self.k:][::-1]
            for idx in top_indices:
                if bm25_scores[idx] > self.bm25_threshold:
                    bm25_docs.append((self.chunk_texts[idx], idx))

        # RRF融合，保留元数据
        fused_docs = self._rrf_fusion(vector_results, bm25_docs)
        return fused_docs

    def _rrf_fusion(self, vec_docs: list[Document], bm25_docs: list[tuple[str, int | None]]):
        """倒数排名融合 (Reciprocal Rank Fusion)，保留 Document 的 metadata"""
        scores = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for i, doc in enumerate(vec_docs):
            key = doc.page_content
            scores[key] += 1 / (60 + i + 1)
            doc_map[key] = doc

        for i, (content, _idx) in enumerate(bm25_docs):
            scores[content] += 1 / (60 + i + 1)
            if content not in doc_map:
                doc_map[content] = Document(page_content=content)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs[:self.k]]
