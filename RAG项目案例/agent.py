from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from datetime import datetime
import config_data as config
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv()


@tool
def get_current_time() -> str:
    """获取当前日期和时间。当你不知道今天是几号、现在几点时，必须使用此工具查询。不需要任何输入参数。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M')}"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。当用户询问实时新闻、天气、股价等实时信息时使用。query 是搜索关键词。"""
    try:
        from ddgs import DDGS

        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        if results:
            formatted = []
            for r in results:
                title = r.get('title', '')
                link = r.get('href') or r.get('url', '')
                formatted.append(f"- {title}: {link}")
            return "\n".join(formatted)
        return "未找到相关信息"
    except Exception as e:
        return f"网络搜索失败: {str(e)}"


# 基础工具列表
BASE_TOOLS = [get_current_time, web_search]


# ==================== DeepSeek Agent ====================

class AgentService:
    def __init__(self, max_iterations=5, vector_store=None):
        """初始化 Agent
        Args:
            max_iterations: 最大迭代次数
            vector_store: 可选的向量存储，如果提供则会自动添加 RAG 工具
        """
        # 初始化 LLM
        self.chat_model = ChatDeepSeek(model=config.chat_model_name)
        self.max_iterations = max_iterations
        self.vector_store = vector_store

        # 构建工具列表
        self.tools = BASE_TOOLS.copy()
        if vector_store:
            # 动态创建 RAG 工具
            self._create_rag_tool()

        self.tool_names = [t.name for t in self.tools]

        # 绑定工具到 LLM
        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

    def _create_rag_tool(self):
        """创建 RAG 工具"""
        from vector_stores import HybridRetriever

        retriever = HybridRetriever(self.vector_store, k=3)

        @tool
        def search_knowledge_base(query: str) -> str:
            """搜索知识库获取相关信息。当用户问技术问题、文档相关内容时使用。query 是搜索关键词。"""
            try:
                docs = retriever.invoke(query)
                if not docs:
                    return "知识库中没有找到相关内容"
                results = []
                for i, doc in enumerate(docs[:3], 1):
                    results.append(f"相关文档 {i}: {doc.page_content[:200]}...")
                return "\n\n".join(results)
            except Exception as e:
                return f"知识库搜索失败: {str(e)}"

        self.tools.append(search_knowledge_base)
        # 更新绑定的工具
        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行工具调用"""
        for t in self.tools:
            if t.name == tool_name:
                try:
                    result = t.invoke(tool_args)
                    return str(result)
                except Exception as e:
                    return f"工具执行失败: {str(e)}"
        return f"未知工具: {tool_name}"

    def invoke(self, query: str, history: str = "") -> str:
        """调用 Agent
        Args:
            query: 用户问题
            history: 对话历史（可选）
        """
        from langchain_core.messages import ToolMessage

        # 构建对话上下文
        if history:
            system_msg = f"""你是一个智能问答助手。当用户问时间、日期时，必须使用 get_current_time 工具获取时间。
当用户问知识库相关问题时，必须使用 search_knowledge_base 工具搜索知识库。

【历史对话】
{history}

你可以直接回答问题，如果需要使用工具，请调用相关工具。"""
        else:
            system_msg = """你是一个智能问答助手。当用户问时间、日期时，必须使用 get_current_time 工具获取时间。
当用户问知识库相关问题时，必须使用 search_knowledge_base 工具搜索知识库。

你可以直接回答问题，如果需要使用工具，请调用相关工具。"""

        # 第一次调用，让模型决定是否需要使用工具
        messages = [
            ("system", system_msg),
            ("user", query)
        ]

        for i in range(self.max_iterations):
            try:
                # 调用 LLM（已绑定工具）
                response = self.llm_with_tools.invoke(messages)

                # 检查是否有工具调用
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # 将 AI 响应添加到消息
                    messages.append(response)

                    # 执行工具调用
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', '')

                        # 执行工具
                        tool_result = self._execute_tool(tool_name, tool_args)

                        # 添加 ToolMessage，需要 tool_call_id
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id
                        ))

                    # 继续循环，让模型根据工具结果生成最终回答
                    continue
                else:
                    # 没有工具调用，直接返回回答
                    return response.content

            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"执行出错: {str(e)}"

        # 达到最大迭代次数，尝试获取最后一条消息
        if messages:
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    return msg.content
                if isinstance(msg, tuple) and msg[1]:
                    return msg[1]

        return f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"


if __name__ == "__main__":
    agent = AgentService()
    print(agent.invoke("今天几号？"))
