from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from datetime import datetime
import config_data as config
from zoneinfo import ZoneInfo
import logging
import hashlib

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


BASE_TOOLS = [get_current_time, web_search]


class AgentService:
    def __init__(self, max_iterations=5, vector_store=None, enable_cache=True, max_retries=2):
        self.chat_model = ChatDeepSeek(model=config.chat_model_name)
        self.max_iterations = max_iterations
        self.vector_store = vector_store
        self.enable_cache = enable_cache
        self.max_retries = max_retries

        self.tools = BASE_TOOLS.copy()
        if vector_store:
            self._create_rag_tool()

        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

        self._cache = {}
        self._max_cache_size = 512  # 缓存上限，防止内存无限增长

    def _create_rag_tool(self):
        from services.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(self.vector_store, k=config.top_k)

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
        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

    def _build_system_prompt(self, history: str = "") -> str:
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

    def _get_cache_key(self, query: str, history: str = "") -> str:
        key_str = f"{query}:{history}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _execute_tool(self, tool_name: str, tool_args: dict, retry: int = None) -> str:
        import json

        if retry is None:
            retry = self.max_retries

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except:
                tool_args = {}

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
        """调用 Agent（非流式）"""
        from langchain_core.messages import ToolMessage

        if self.enable_cache:
            cache_key = self._get_cache_key(query, history)
            if cache_key in self._cache:
                logger.info(f"命中缓存: {query[:20]}...")
                return self._cache[cache_key]

        system_msg = self._build_system_prompt(history)

        messages = [
            ("system", system_msg),
            ("user", query)
        ]

        for i in range(self.max_iterations):
            try:
                response = self.llm_with_tools.invoke(messages)

                if hasattr(response, 'tool_calls') and response.tool_calls:
                    messages.append(response)

                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})
                        tool_call_id = tool_call.get('id', '')

                        tool_result = self._execute_tool(tool_name, tool_args)

                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id
                        ))

                    logger.info(f"第 {i + 1} 轮：调用了工具，继续推理")
                    continue
                else:
                    result = response.content

                    if self.enable_cache:
                        self._add_to_cache(cache_key, result)

                    return result

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Agent 执行出错: {e}")
                return f"执行出错: {str(e)}"

        if messages:
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    result = msg.content
                    if self.enable_cache:
                        self._add_to_cache(cache_key, result)
                    return result
                if isinstance(msg, tuple) and msg[1]:
                    result = msg[1]
                    if self.enable_cache:
                        self._add_to_cache(cache_key, result)
                    return result

        return f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"

    def _add_to_cache(self, key: str, value: str):
        """写入缓存，超过上限时淘汰最旧条目"""
        if len(self._cache) >= self._max_cache_size:
            # 删除最早添加的条目（FIFO）
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")

    def stream_events(self, query: str, history: str = ""):
        """
        真正的流式 Agent — 边生成边产出 token，工具调用后继续流式整理回复。

        产出格式:
          {"type": "tool_start", "tool_name": str, "args": dict}
          {"type": "tool_end",   "tool_name": str, "result": str}
          {"type": "token",      "content": str}
          {"type": "done",       "full_response": str}
          {"type": "error",      "error": str}
        """
        import json
        from langchain_core.messages import ToolMessage, AIMessage

        system_msg = self._build_system_prompt(history)

        messages = [
            ("system", system_msg),
            ("user", query)
        ]

        for iteration in range(self.max_iterations):
            try:
                full_content = ""
                tool_call_chunks: dict[int, dict] = {}

                # 真正的流式调用 — 每个 chunk 产出 token
                # 注：bind_tools 返回的 RunnableBinding 流式输出 AIMessageChunk，
                #     chunk 本身就是消息对象，没有 .message 包装层
                for chunk in self.llm_with_tools.stream(messages):
                    # 流式输出文本（chunk 即 AIMessageChunk）
                    if chunk.content:
                        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        full_content += text
                        yield {"type": "token", "content": text}

                    # 累积 tool_call 片段
                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            idx = tc.get('index', 0)
                            if idx not in tool_call_chunks:
                                tool_call_chunks[idx] = {'name': '', 'args': '', 'id': ''}
                            if tc.get('name'):
                                tool_call_chunks[idx]['name'] += tc['name']
                            if tc.get('args'):
                                tool_call_chunks[idx]['args'] += tc['args']
                            if tc.get('id') and not tool_call_chunks[idx]['id']:
                                tool_call_chunks[idx]['id'] = tc['id']

                # 流结束后检查是否有工具调用
                if tool_call_chunks:
                    complete_tool_calls = []
                    for idx in sorted(tool_call_chunks.keys()):
                        tc = tool_call_chunks[idx]
                        try:
                            args = json.loads(tc['args']) if tc['args'] else {}
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        complete_tool_calls.append({
                            'name': tc['name'],
                            'args': args,
                            'id': tc['id'] or str(idx),
                        })

                    # 将 AI 消息（含工具调用）写入历史
                    messages.append(AIMessage(
                        content=full_content,
                        tool_calls=complete_tool_calls
                    ))

                    # 执行工具并产出事件
                    for tc_data in complete_tool_calls:
                        yield {"type": "tool_start", "tool_name": tc_data['name'], "args": tc_data['args']}

                        tool_result = self._execute_tool(tc_data['name'], tc_data['args'])

                        yield {"type": "tool_end", "tool_name": tc_data['name'], "result": tool_result}

                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tc_data['id']
                        ))

                    logger.info(f"第 {iteration + 1} 轮：调用了 {len(complete_tool_calls)} 个工具，继续流式整理回复...")
                    # 继续循环，让 LLM 基于工具结果流式生成最终回复
                    continue
                else:
                    # 没有工具调用 — 流式输出完毕
                    yield {"type": "done", "full_response": full_content}
                    return

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Agent 流式执行出错: {e}")
                yield {"type": "error", "error": str(e)}
                return

        yield {"type": "error", "error": f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"}

    def generate_suggestions(self, query: str, response: str, history: str = "") -> list:
        """生成追问建议"""
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

            cleaned = []
            for s in suggestions[:3]:
                s = s.strip()
                s = s.lstrip('123456789.、) ')
                if s and len(s) <= 20:
                    cleaned.append(s)

            return cleaned[:3] if cleaned else ["谢谢", "还有其他问题吗", "可以详细说说吗"]

        except Exception as e:
            logger.warning(f"生成追问建议失败: {e}")
            return ["谢谢", "还有其他问题吗", "可以详细说说吗"]
