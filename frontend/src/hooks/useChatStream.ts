import { useCallback, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { streamChat } from '../api/client';
import { apiPost } from '../api/client';
import type { ToolCallRecord } from '../types/chat';

export function useChatStream() {
  const abortRef = useRef<AbortController | null>(null);
  const addUserMessage = useChatStore((s) => s.addUserMessage);
  const startStreaming = useChatStore((s) => s.startStreaming);
  const setToolCall = useChatStore((s) => s.setToolCall);
  const appendToken = useChatStore((s) => s.appendToken);
  const finishStreaming = useChatStore((s) => s.finishStreaming);
  const setSuggestions = useChatStore((s) => s.setSuggestions);
  const isLoading = useChatStore((s) => s.isLoading);

  const sendMessage = useCallback(
    async (message: string) => {
      const { sessionId, isLoading } = useChatStore.getState();
      if (!sessionId || isLoading) return;

      addUserMessage(message);
      startStreaming();

      const abortController = new AbortController();
      abortRef.current = abortController;

      const toolCalls: ToolCallRecord[] = [];

      try {
        for await (const event of streamChat(sessionId, message, abortController.signal)) {
          switch (event.event) {
            case 'tool_start': {
              const td = event.data as { tool_name: string; args: Record<string, unknown> };
              const tc: ToolCallRecord = { toolName: td.tool_name, args: td.args, status: 'running' };
              toolCalls.push(tc);
              setToolCall(tc);
              break;
            }
            case 'tool_end': {
              const td = event.data as { tool_name: string; result: string };
              const existing = toolCalls.find((t) => t.toolName === td.tool_name && t.status === 'running');
              if (existing) {
                existing.result = td.result;
                existing.status = 'done';
                setToolCall({ ...existing });
              }
              break;
            }
            case 'token': {
              const td = event.data as { content: string };
              appendToken(td.content);
              break;
            }
            case 'done':
              break;
            case 'error': {
              const ed = event.data as { error: string };
              console.error('Stream error:', ed.error);
              break;
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // 用户主动取消
        } else {
          console.error('Chat error:', err);
        }
      }

      // 用流式内容完成消息
      const { streamingContent } = useChatStore.getState();
      if (streamingContent) {
        // 为最后一条 assistant 消息附加 toolCalls
        const msg = {
          id: `msg_${Date.now()}`,
          role: 'assistant' as const,
          content: streamingContent,
          toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        };
        useChatStore.setState((s) => ({
          messages: [...s.messages, msg],
          isLoading: false,
          streamingContent: '',
          activeToolCall: null,
        }));
      } else {
        finishStreaming('');
      }

      abortRef.current = null;

      // 获取追问建议
      try {
        const { sessionId: sid, messages: msgs } = useChatStore.getState();
        if (msgs.length >= 2) {
          const lastUser = [...msgs].reverse().find((m) => m.role === 'user');
          const lastAssistant = [...msgs].reverse().find((m) => m.role === 'assistant');
          if (lastUser && lastAssistant) {
            const res = await apiPost<{ suggestions: string[] }>('/api/chat/suggestions', {
              session_id: sid,
              query: lastUser.content,
              response: lastAssistant.content,
            });
            setSuggestions(res.suggestions);
          }
        }
      } catch {
        // 静默失败
      }
    },
    [addUserMessage, startStreaming, setToolCall, appendToken, finishStreaming, setSuggestions]
  );

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { sendMessage, stopGeneration, isLoading };
}
