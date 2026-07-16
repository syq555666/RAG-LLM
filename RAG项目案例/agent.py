from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from datetime import datetime
import config_data as config
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import logging
import hashlib

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        logger.error(f"网络搜索失败: {e}")
        return f"网络搜索失败: {str(e)}"


# 基础工具列表
BASE_TOOLS = [get_current_time, web_search]


# ==================== Agent 优化版本 ====================

class AgentService:
    def __init__(self, max_iterations=5, vector_store=None, enable_cache=True, max_retries=2):
        """初始化 Agent
        Args:
            max_iterations: 最大迭代次数
            vector_store: 可选的向量存储，如果提供则会自动添加 RAG 工具
            enable_cache: 是否启用结果缓存
            max_retries: 工具执行最大重试次数
        """
        # 初始化 LLM
        self.chat_model = ChatDeepSeek(model=config.chat_model_name)
        self.max_iterations = max_iterations
        self.vector_store = vector_store
        self.enable_cache = enable_cache
        self.max_retries = max_retries

        # 构建工具列表
        self.tools = BASE_TOOLS.copy()
        if vector_store:
            # 动态创建 RAG 工具
            self._create_rag_tool()

        self.tool_names = [t.name for t in self.tools]

        # 绑定工具到 LLM
        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

        # 意图识别 LLM（不绑定工具）
        self.intent_llm = ChatDeepSeek(
            model=config.chat_model_name,
            temperature=0
        )

        # 缓存字典
        self._cache = {}

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
                logger.error(f"知识库搜索失败: {e}")
                return f"知识库搜索失败: {str(e)}"

        self.tools.append(search_knowledge_base)
        # 更新绑定的工具
        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

    def _build_system_prompt(self, history: str = "") -> str:
        """构建增强的系统提示词"""
        base_prompt = """你是一个专业的智能客服助手。

## 你的能力
1. 回答用户问题
2. 使用工具获取实时信息
3. 搜索知识库获取相关文档

## 回答要求
1. 回答要简洁明了，用 Markdown 格式组织回答
2. 如果知识库没有找到相关信息，请明确告知用户
3. 如果需要使用工具，必须调用相关工具，不要假设结果
4. 不要重复调用同一个工具
5. 如果不确定信息，请如实告知用户"""

        tool_instructions = """
## 工具使用规则
- 当用户问时间、日期时 → 使用 get_current_time
- 当用户问知识库相关问题时 → 使用 search_knowledge_base
- 当用户问实时新闻、天气、股价等 → 使用 web_search
- 其他问题可以直接回答"""

        if history:
            return f"""{base_prompt}

{tool_instructions}

【历史对话】
{history}

现在开始回答用户问题。"""
        else:
            return f"""{base_prompt}

{tool_instructions}

现在开始回答用户问题。"""

    def _intent_recognition(self, query: str) -> dict:
        """意图识别：判断问题类型和是否需要工具"""
        intent_prompt = f"""请分析用户问题，判断是否需要调用工具。

问题：{query}

请从以下类别中选择：
1. time - 询问时间、日期
2. knowledge - 询问知识库相关问题（产品信息、技术文档等）
3. realtime - 询问实时信息（新闻、天气、股价等）
4. general - 一般问题，不需要工具

直接输出类别名称，不要有其他内容。"""

        try:
            response = self.intent_llm.invoke(intent_prompt)
            intent = response.content.strip().lower()

            # 映射到工具
            intent_map = {
                'time': ['get_current_time'],
                'knowledge': ['search_knowledge_base'],
                'realtime': ['web_search'],
                'general': []
            }

            needed_tools = []
            for key, tools in intent_map.items():
                if key in intent:
                    needed_tools.extend(tools)
                    break
            else:
                # 默认认为可能需要工具，让模型自己判断
                needed_tools = []

            logger.info(f"意图识别结果: {intent}, 建议工具: {needed_tools}")
            return {'intent': intent, 'needed_tools': needed_tools}

        except Exception as e:
            logger.warning(f"意图识别失败: {e}，使用默认策略")
            return {'intent': 'unknown', 'needed_tools': []}

    def _get_cache_key(self, query: str, history: str = "") -> str:
        """生成缓存 key"""
        key_str = f"{query}:{history}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _execute_tool(self, tool_name: str, tool_args: dict, retry: int = None) -> str:
        """执行工具调用（带重试机制）"""
        import json

        if retry is None:
            retry = self.max_retries

        # 处理参数格式：可能是 dict 或 JSON 字符串
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except:
                tool_args = {}

        # 确保 tool_args 是字典
        if not isinstance(tool_args, dict):
            tool_args = {}

        for attempt in range(retry):
            for t in self.tools:
                if t.name == tool_name:
                    try:
                        logger.info(f"执行工具: {tool_name}, 参数: {tool_args}, 尝试: {attempt + 1}/{retry}")
                        result = t.invoke(tool_args)
                        logger.info(f"工具执行成功: {tool_name}")
                        return str(result)
                    except Exception as e:
                        logger.warning(f"工具执行失败: {tool_name}, 错误: {e}, 尝试: {attempt + 1}/{retry}")
                        if attempt == retry - 1:
                            return f"工具执行失败: {str(e)}"
        return f"未知工具: {tool_name}"

    def invoke(self, query: str, history: str = "") -> str:
        """调用 Agent
        Args:
            query: 用户问题
            history: 对话历史（可选）
        """
        from langchain_core.messages import ToolMessage

        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(query, history)
            if cache_key in self._cache:
                logger.info(f"命中缓存: {query[:20]}...")
                return self._cache[cache_key]

        # 意图识别（可选，可以跳过以提高速度）
        # intent_result = self._intent_recognition(query)

        # 构建对话上下文
        system_msg = self._build_system_prompt(history)

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

                        # 执行工具（带重试）
                        tool_result = self._execute_tool(tool_name, tool_args)

                        # 添加 ToolMessage，需要 tool_call_id
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id
                        ))

                    # 继续循环，让模型根据工具结果生成最终回答
                    logger.info(f"第 {i + 1} 轮：调用了工具，继续推理")
                    continue
                else:
                    # 没有工具调用，直接返回回答
                    result = response.content

                    # 存入缓存
                    if self.enable_cache:
                        self._cache[cache_key] = result

                    return result

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Agent 执行出错: {e}")
                return f"执行出错: {str(e)}"

        # 达到最大迭代次数，尝试获取最后一条消息
        if messages:
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    result = msg.content

                    # 存入缓存
                    if self.enable_cache:
                        self._cache[cache_key] = result

                    return result
                if isinstance(msg, tuple) and msg[1]:
                    result = msg[1]

                    # 存入缓存
                    if self.enable_cache:
                        self._cache[cache_key] = result

                    return result

        return f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")

    def stream(self, query: str, history: str = "", tool_callback=None):
        """流式调用 Agent（生成器）
        先执行工具调用，最后流式输出结果
        Args:
            query: 用户问题
            history: 对话历史（可选）
            tool_callback: 工具执行回调函数
        Yields:
            str: 流式输出的文本片段
        """
        from langchain_core.messages import ToolMessage

        # 构建对话上下文
        system_msg = self._build_system_prompt(history)

        messages = [
            ("system", system_msg),
            ("user", query)
        ]

        for _ in range(self.max_iterations):
            try:
                # 调用 LLM
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

                        # 通知即将执行工具
                        if tool_callback:
                            tool_callback(tool_name, tool_args)

                        tool_result = self._execute_tool(tool_name, tool_args)

                        # 工具执行完成
                        if tool_callback:
                            tool_callback(tool_name, tool_args, result=tool_result)

                        # 添加 ToolMessage
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id
                        ))

                    # 继续循环
                    continue
                else:
                    # 没有工具调用，获取最终回答
                    final_response = response.content
                    break

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Agent 流式执行出错: {e}")
                yield f"执行出错: {str(e)}"
                return

        else:
            # 达到最大迭代次数
            yield f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"
            return

        # 流式输出最终回答
        for char in final_response:
            yield char

    def generate_suggestions(self, query: str, response: str, history: str = "") -> list:
        """生成追问建议
        Args:
            query: 用户问题
            response: AI 回答
            history: 对话历史
        Returns:
            list: 3 个建议问题
        """
        context = f"\n历史对话：{history}" if history else ""
        suggestion_prompt = f"""基于以下对话，生成 3 个用户可能会追问的相关问题。

用户问题：{query}
AI 回答：{response}{context}

要求：
1. 生成 3 个简洁的相关问题
2. 每个问题不超过 20 个字
3. 只输出问题，不要有其他内容
4. 用中文"""

        try:
            result = self.chat_model.invoke(suggestion_prompt)
            suggestions = result.content.strip().split('\n')

            # 清理建议，提取问题
            cleaned = []
            for s in suggestions[:3]:
                s = s.strip()
                # 去除可能的序号
                s = s.lstrip('123456789.、) ')
                if s and len(s) <= 20:
                    cleaned.append(s)

            return cleaned[:3] if cleaned else ["谢谢", "还有其他问题吗", "可以详细说说吗"]

        except Exception as e:
            logger.warning(f"生成追问建议失败: {e}")
            return ["谢谢", "还有其他问题吗", "可以详细说说吗"]


if __name__ == "__main__":
    agent = AgentService()
    print(agent.invoke("今天几号？"))
    print(agent.invoke("你们公司是做什么的？"))
