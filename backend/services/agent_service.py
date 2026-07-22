"""Agent 服务 — 核心推理引擎（流式 + 非流式）"""

import json
import time
import hashlib
import logging
from langchain_core.messages import ToolMessage, AIMessage
from langchain_deepseek import ChatDeepSeek

from services.prompts import build_system_prompt, build_suggestions_prompt
from services.tools import BASE_TOOLS, create_rag_tool
from services.hybrid_retriever import HybridRetriever
import config_data as config

logger = logging.getLogger(__name__)

# 缓存 TTL（秒）
CACHE_TTL_SECONDS = 300  # 5 分钟


class AgentService:
    def __init__(
        self,
        max_iterations=5,
        vector_store=None,
        enable_cache=True,
        max_retries=2,
        chat_model: ChatDeepSeek | None = None,
    ):
        # 使用传入的 LLM 实例或创建新的（向后兼容）
        self.chat_model = chat_model or ChatDeepSeek(model=config.chat_model_name)
        self.max_iterations = max_iterations
        self.vector_store = vector_store
        self.enable_cache = enable_cache
        self.max_retries = max_retries

        # 初始化工具列表和 O(1) 字典查找
        self.tools = list(BASE_TOOLS)  # 复制一份
        self._tool_map: dict[str, object] = {t.name: t for t in self.tools}
        self._retriever: HybridRetriever | None = None

        if vector_store:
            self._create_rag_tool()

        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

        # 带 TTL 的缓存: {key: (value, timestamp)}
        self._cache: dict[str, tuple[str, float]] = {}
        self._max_cache_size = 512

    def _create_rag_tool(self):
        """创建并注册知识库搜索工具"""
        self._retriever = HybridRetriever(self.vector_store, k=config.top_k)
        rag_tool = create_rag_tool(self._retriever)
        self.tools.append(rag_tool)
        self._tool_map[rag_tool.name] = rag_tool
        self.llm_with_tools = self.chat_model.bind_tools(self.tools)

    def invalidate_retriever(self):
        """知识库变更后失效检索器索引（由 deps.py 回调触发）"""
        if self._retriever:
            self._retriever.invalidate()
        # 同时清空缓存，防止返回基于旧知识库的答案
        if self.enable_cache:
            self.clear_cache()

    def _get_cache_key(self, query: str, history: str = "") -> str:
        key_str = f"{query}:{history}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        """从缓存获取，检查 TTL"""
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > CACHE_TTL_SECONDS:
            del self._cache[key]
            return None
        return value

    def _cache_set(self, key: str, value: str):
        """写入缓存，超过上限时 FIFO 淘汰"""
        if len(self._cache) >= self._max_cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = (value, time.time())

    def _execute_tool(self, tool_name: str, tool_args: dict, attempts: int = None) -> str:
        """执行工具，O(1) 查找 + 重试"""
        if attempts is None:
            attempts = self.max_retries

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

        if not isinstance(tool_args, dict):
            tool_args = {}

        tool = self._tool_map.get(tool_name)
        if tool is None:
            return f"未知工具: {tool_name}"

        for attempt in range(attempts):
            try:
                logger.info(f"执行工具: {tool_name}, 参数: {tool_args}, 尝试: {attempt + 1}/{attempts}")
                result = tool.invoke(tool_args)
                logger.info(f"工具执行成功: {tool_name}")
                return str(result)
            except Exception as e:
                logger.warning(f"工具执行失败: {tool_name}, 错误: {e}, 尝试: {attempt + 1}/{attempts}")
                if attempt == attempts - 1:
                    return f"工具执行失败: {str(e)}"
        return f"工具执行失败: {tool_name}"

    def invoke(self, query: str, history: str = "") -> str:
        """调用 Agent（非流式）"""
        if self.enable_cache:
            cache_key = self._get_cache_key(query, history)
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.info(f"命中缓存: {query[:20]}...")
                return cached

        system_msg = build_system_prompt(history)
        messages = [
            ("system", system_msg),
            ("user", query)
        ]
        last_response = ""

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
                    last_response = response.content

                    if self.enable_cache:
                        self._cache_set(cache_key, last_response)

                    return last_response

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Agent 执行出错: {e}")
                return f"执行出错: {str(e)}"

        return last_response or f"已达到最大迭代次数({self.max_iterations})，未能得到答案。"

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Agent 缓存已清空")

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
        system_msg = build_system_prompt(history)

        messages = [
            ("system", system_msg),
            ("user", query)
        ]

        for iteration in range(self.max_iterations):
            try:
                full_content = ""
                tool_call_chunks: dict[int, dict] = {}

                # 真正的流式调用 — 每个 chunk 产出 token
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
        suggestion_prompt = build_suggestions_prompt(query, response, history)

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
