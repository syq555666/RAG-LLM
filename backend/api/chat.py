"""聊天 API — SSE 流式对话 + 追问建议"""

import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from api.deps import get_agent_service, get_history_store
from schemas.chat import StreamChatRequest, SuggestionsRequest, SuggestionsResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(request: StreamChatRequest):
    """SSE 流式对话端点

    使用 asyncio.Queue 桥接同步 Agent 生成器和异步 SSE 响应，
    仅一次 run_in_executor，避免逐 token 线程调度开销。
    """
    agent = get_agent_service()
    history_store = get_history_store(request.session_id)
    history_str = history_store.get_context_for_llm()

    # 保存用户消息
    history_store.add_messages([HumanMessage(content=request.message)])

    async def event_generator():
        full_response = ""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            """在独立线程中运行同步生成器，将事件推入异步队列"""
            try:
                for event in agent.stream_events(request.message, history=history_str):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                # 哨兵：表示生成完毕
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "error": str(e)}
                )

        # 在线程池中启动 producer（仅这一次线程调度！）
        loop.run_in_executor(None, produce)

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break  # 生成完毕

                event_type = event.get("type", "")

                if event_type == "token":
                    full_response += event["content"]
                    yield f"event: token\ndata: {json.dumps({'content': event['content']}, ensure_ascii=False)}\n\n"

                elif event_type == "tool_start":
                    yield f"event: tool_start\ndata: {json.dumps({'tool_name': event['tool_name'], 'args': event.get('args', {})}, ensure_ascii=False)}\n\n"

                elif event_type == "tool_end":
                    yield f"event: tool_end\ndata: {json.dumps({'tool_name': event['tool_name'], 'result': event['result']}, ensure_ascii=False)}\n\n"

                elif event_type == "done":
                    full_response = event.get("full_response", full_response)
                    # done 事件由 producer 发送哨兵 None 后自然结束
                    # 这里只是捕获 full_response

                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps({'error': event['error']}, ensure_ascii=False)}\n\n"
                    return

            # 保存 AI 回答
            if full_response:
                history_store.add_messages([AIMessage(content=full_response)])

            yield f"event: done\ndata: {json.dumps({'full_response': full_response}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(request: SuggestionsRequest):
    """获取追问建议"""
    agent = get_agent_service()
    history_store = get_history_store(request.session_id)
    history_str = history_store.get_context_for_llm()

    loop = asyncio.get_running_loop()
    suggestions = await loop.run_in_executor(
        None,
        lambda: agent.generate_suggestions(request.query, request.response, history=history_str)
    )
    return SuggestionsResponse(suggestions=suggestions)
