import asyncio
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from api.deps import get_agent_service, get_history_store
from schemas.chat import StreamChatRequest, SuggestionsRequest, SuggestionsResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def stream_chat(request: StreamChatRequest):
    """SSE 流式对话端点"""
    agent = get_agent_service()
    history_store = get_history_store(request.session_id)
    history_str = history_store.get_context_for_llm()

    # 保存用户消息
    history_store.add_messages([HumanMessage(content=request.message)])

    async def event_generator():
        full_response = ""
        loop = asyncio.get_event_loop()

        try:
            # 在线程池中运行同步的 agent 流式生成
            gen = agent.stream_events(request.message, history=history_str)

            while True:
                # 在线程池中获取下一个事件（避免阻塞事件循环）
                event = await loop.run_in_executor(None, lambda g=gen: next(g, None))
                if event is None:
                    break

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
                    break

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
def get_suggestions(request: SuggestionsRequest):
    """获取追问建议"""
    agent = get_agent_service()
    history_store = get_history_store(request.session_id)
    history_str = history_store.get_context_for_llm()

    suggestions = agent.generate_suggestions(request.query, request.response, history=history_str)
    return SuggestionsResponse(suggestions=suggestions)
