from langchain_chroma import Chroma
from langchain_core.documents import Document
import config_data as config
import os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")


class HybridRetriever:
    """混合检索器：结合向量检索和BM25关键词检索"""

    def __init__(self, vector_store, k: int = 5):
        self.vector_store = vector_store
        self.k = k
        self.bm25 = None
        self.chunk_texts = []
        self._index_built = False

    def _build_bm25_index(self):
        """从Chroma获取所有文档构建BM25索引"""
        all_data = self.vector_store.get()
        documents = all_data.get("documents", [])

        if documents:
            self.chunk_texts = documents
            tokenized_corpus = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._index_built = True

    def invoke(self, query: str) -> list[Document]:
        """执行混合检索，返回融合后的文档"""
        if not self._index_built:
            self._build_bm25_index()

        # 向量检索
        vector_results = self.vector_store.similarity_search(query, k=self.k)
        vector_docs = [doc.page_content for doc in vector_results]

        # BM25检索
        bm25_docs = []
        if self._index_built and self.bm25:
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(bm25_scores)[-self.k:][::-1]
            bm25_docs = [self.chunk_texts[i] for i in top_indices if bm25_scores[i] > 0]

        # RRF融合
        fused_docs = self._rrf_fusion(vector_docs, bm25_docs)
        return [Document(page_content=doc) for doc in fused_docs]

    def _rrf_fusion(self, vec_docs: list, bm25_docs: list):
        """倒数排名融合 (Reciprocal Rank Fusion)"""
        scores = defaultdict(float)

        for i, doc in enumerate(vec_docs):
            scores[doc] += 1 / (60 + i + 1)
        for i, doc in enumerate(bm25_docs):
            scores[doc] += 1 / (60 + i + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:self.k]]


class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name = config.collection_name,
            embedding_function = self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.top_k})


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)