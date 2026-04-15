from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from vector_stores import VectorStoreService, HybridRetriever
import config_data as config
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from file_history_store import get_history
from prompts import RAG_PROMPT_TEMPLATE

load_dotenv()


class RagService:
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )
        # 使用混合检索器
        self.hybrid_retriever = HybridRetriever(self.vector_service.vector_store, k=5)

        # 使用独立的 prompt 模板文件
        self.prompt_template = RAG_PROMPT_TEMPLATE

        self.chat_model = ChatTongyi(model = config.chat_model_name)

        self.chain = self.__get_chain()


    def __get_chain(self):
        def safe_retriever(query):
            """安全检索，处理空知识库的情况"""
            try:
                return self.hybrid_retriever.invoke(query)
            except Exception:
                return []

        # 使用安全检索器
        retriever = RunnableLambda(safe_retriever)

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str


        def format_for_retriever(value):
            # value 已经是 {"input": prompt} 格式
            return value.get("input", "")

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]

            return new_value


        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | RunnableLambda(format_document),
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | self.chat_model | StrOutputParser()
        )


        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain