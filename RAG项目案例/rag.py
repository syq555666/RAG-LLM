from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from vector_stores import VectorStoreService, HybridRetriever
import config_data as config
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
from file_history_store import get_history

load_dotenv()


def debug_print_prompt(prompt):
    """调试模式下打印 prompt"""
    if config.debug_mode:
        print("=" * 20)
        print(prompt.to_string())
        print("=" * 20)
    return prompt

class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )
        # 使用混合检索器
        self.hybrid_retriever = HybridRetriever(self.vector_service.vector_store, k=5)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个专业的客服助手。请根据以下参考资料来回答用户的问题。"
                 "\n\n【参考资料】\n{context}"),
                ("system", "以下是与用户的对话历史："),
                MessagesPlaceholder("history"),
                ("user", "{input}")
            ]
        )

        self.chat_model = ChatTongyi(model = config.chat_model_name)

        self.chain = self.__get_chain()


    def __get_chain(self):
        # 使用混合检索器，用 RunnableLambda 包装
        from langchain_core.runnables import RunnableLambda
        retriever = RunnableLambda(lambda q: self.hybrid_retriever.invoke(q))

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str


        def format_for_retriever(value : dict) -> str:
            return value["input"]

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
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | debug_print_prompt | self.chat_model | StrOutputParser()
        )


        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "羽绒服怎么清洗"}, session_config)
    print(res)