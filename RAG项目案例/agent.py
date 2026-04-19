from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
import config_data as config
import re


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


# 工具列表
TOOLS = [web_search]
TOOL_NAMES = [t.name for t in TOOLS]
TOOL_DESCRIPTIONS = "\n".join([f"- {t.name}: {t.description}" for t in TOOLS])


# ==================== ReAct Agent ====================

class AgentService:
    def __init__(self, max_iterations=5):
        # 初始化 LLM
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.max_iterations = max_iterations

        # 初始化向量服务（避免每次调用都重新创建）
        from langchain_community.embeddings import DashScopeEmbeddings
        from vector_stores import VectorStoreService
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )

        # ReAct Prompt
        self.react_prompt = ChatPromptTemplate.from_template("""你是一个智能助手，可以使用工具来回答问题。

可用工具：
{tool_descriptions}

你必须按照以下格式思考和行动：

问题: {input}
{agent_scratchpad}
思考: 分析这个问题，决定下一步做什么
行动: 工具名称（必须是 {tool_names} 之一）
行动输入: 给工具的输入
观察: 工具执行的结果
... (这个过程可以重复多次)
思考: 我现在有足够的信息可以回答问题了
最终答案: 给用户的回答

现在开始！

问题: {input}
{agent_scratchpad}
思考: 分析这个问题，决定下一步做什么
行动:""")

        self.prompt_template = self.react_prompt.partial(
            tool_descriptions=TOOL_DESCRIPTIONS,
            tool_names=", ".join(TOOL_NAMES)
        )

    def _search_knowledge_base(self, query: str) -> str:
        """搜索知识库"""
        try:
            from vector_stores import HybridRetriever

            retriever = HybridRetriever(self.vector_service.vector_store, k=3)
            docs = retriever.invoke(query)

            if not docs:
                return "知识库中没有找到相关内容"

            results = []
            for i, doc in enumerate(docs[:3], 1):
                results.append(f"相关文档 {i}: {doc.page_content[:200]}...")

            return "\n\n".join(results)
        except Exception as e:
            return f"知识库搜索失败: {str(e)}"

    def _parse_response(self, response: str) -> dict:
        """解析模型输出，提取行动和观察"""
        if "最终答案:" in response:
            answer = response.split("最终答案:")[-1].strip()
            return {"action": None, "action_input": None, "final_answer": answer, "done": True}

        action = None
        action_input = None

        action_matches = re.findall(r"行动:\s*(\w+)", response)
        if action_matches:
            action = action_matches[-1].strip()

        if action:
            action_input_match = re.search(rf"行动:\s*{action}\s*\n行动输入:\s*(.+?)(?:\n行动:|$)", response, re.DOTALL)
            if action_input_match:
                action_input = action_input_match.group(1).strip()

        if action is None or action_input is None:
            return {"action": None, "action_input": None, "final_answer": response.strip(), "done": True}

        return {"action": action, "action_input": action_input, "final_answer": None, "done": False}

    def _execute_action(self, action: str, action_input: str) -> str:
        """执行工具调用"""
        if action == "search_knowledge_base":
            result = self._search_knowledge_base(action_input)
            return f"观察: {result}"
        elif action == "web_search":
            result = web_search.invoke(action_input)
            return f"观察: {result}"
        return f"观察: 未知工具 '{action}'"

    def invoke(self, query: str) -> str:
        """调用 ReAct Agent"""
        scratchpad = ""

        for _ in range(self.max_iterations):
            try:
                prompt_input = {"input": query, "agent_scratchpad": scratchpad if scratchpad else ""}
                prompt = self.prompt_template.invoke(prompt_input)
                llm_response = self.chat_model.invoke(prompt).content

                parsed = self._parse_response(llm_response)

                if parsed["done"]:
                    return parsed["final_answer"]

                action = parsed["action"]
                action_input = parsed["action_input"]

                if action and action in TOOL_NAMES:
                    observation = self._execute_action(action, action_input)
                    scratchpad += f"\n思考: 分析问题\n行动: {action}\n行动输入: {action_input}\n{observation}\n思考:"
                else:
                    return llm_response

            except Exception as e:
                return f"执行出错: {str(e)}"

        return f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"


if __name__ == "__main__":
    agent = AgentService()
    print(agent.invoke("今天天气怎么样"))