from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config_data as config
import re


# ==================== 工具定义 ====================

@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库中的相关内容。当用户询问产品信息、公司资料等问题时使用此工具。"""
    try:
        from vector_stores import VectorStoreService, HybridRetriever
        from langchain_community.embeddings import DashScopeEmbeddings
        import config_data as cfg

        vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=cfg.embedding_model_name),
        )
        retriever = HybridRetriever(vector_service.vector_store, k=3)
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中没有找到相关内容"

        results = []
        for i, doc in enumerate(docs[:3], 1):
            results.append(f"相关文档 {i}: {doc.page_content[:200]}...")

        return "\n\n".join(results)
    except Exception as e:
        return f"知识库搜索失败: {str(e)}"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。当用户询问实时新闻、天气、股价等实时信息时使用。"""
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


@tool
def calculator(expression: str) -> str:
    """数学计算器。用于数学计算、金额计算等。只接受数学表达式，如 "2+3*5"、"100/4" 等。"""
    try:
        if not re.match(r'^[\d\+\-\*\/\.\(\)\s]+$', expression):
            return "计算表达式包含非法字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 工具列表
TOOLS = [search_knowledge_base, web_search, calculator]
TOOL_NAMES = {t.name: t for t in TOOLS}


# ==================== Agent 服务 ====================

class AgentService:
    def __init__(self):
        # 初始化 LLM
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        # 创建 Agent
        self._setup_agent()

    def _setup_agent(self):
        """设置 Agent"""
        # 判断是否需要调用工具的 prompt
        self.judge_prompt = ChatPromptTemplate.from_template("""你是一个智能助手。根据用户问题判断需要使用的工具。

可用工具：
- search_knowledge_base: 搜索知识库
- web_search: 搜索互联网
- calculator: 数学计算

用户问题: {input}

请判断需要使用哪个工具（只输出工具名称，不需要其他内容）：
如果不需要工具，请直接回答"无需工具".""")

        # 构造回答的 prompt
        self.answer_prompt = ChatPromptTemplate.from_template("""你是一个智能助手，请根据以下搜索结果回答用户问题。

用户问题: {input}

知识库搜索结果:
{kb_result}

网络搜索结果:
{web_result}

计算结果:
{calc_result}

请根据以上搜索结果回答用户问题。如果网络搜索结果中有相关信息，请整理后回答。如果没有相关信息，请如实说明。
回答要求：直接给出答案，不要提及"根据搜索结果"这类话。""")

        self.judge_chain = self.judge_prompt | self.chat_model | StrOutputParser()
        self.answer_chain = self.answer_prompt | self.chat_model | StrOutputParser()

    def invoke(self, query: str) -> str:
        """调用 Agent"""
        try:
            # 1. 判断需要使用的工具
            tool_name = self.judge_chain.invoke({"input": query}).strip()
            print(f"判断工具: {tool_name}")

            # 2. 初始化结果
            kb_result = "无"
            web_result = "无"
            calc_result = "无"

            # 3. 根据判断调用相应工具（支持多个工具）
            if "无需工具" not in tool_name:
                if "search_knowledge_base" in tool_name:
                    try:
                        kb_result = search_knowledge_base.invoke(query)
                        print(f"知识库结果: {kb_result[:100]}...")
                    except Exception as e:
                        kb_result = f"搜索失败: {e}"

                if "web_search" in tool_name:
                    try:
                        web_result = web_search.invoke(query)
                        print(f"网络结果: {web_result[:100]}...")
                    except Exception as e:
                        web_result = f"搜索失败: {e}"

                if "calculator" in tool_name:
                    # 提取计算表达式
                    expr = re.findall(r'[\d\+\-\*\/\.\(\)\s]+', query)
                    expr = "".join(expr) if expr else None
                    if expr:
                        try:
                            calc_result = calculator.invoke(expr)
                            print(f"计算结果: {calc_result}")
                        except Exception as e:
                            calc_result = f"计算失败: {e}"
                    else:
                        calc_result = "无法从问题中提取计算表达式"

            # 4. 检查是否有有效结果
            has_result = any([kb_result != "无", web_result != "无", calc_result != "无" and "无法提取" not in calc_result])

            if not has_result:
                # 没有调用工具，直接让 LLM 回答
                return self.chat_model.invoke(query).content

            # 5. 生成最终回答
            answer = self.answer_chain.invoke({
                "input": query,
                "kb_result": kb_result,
                "web_result": web_result,
                "calc_result": calc_result
            })

            return answer

        except Exception as e:
            return f"调用失败: {str(e)}"


if __name__ == "__main__":
    agent = AgentService()
    print(agent.invoke("计算 100+200"))
