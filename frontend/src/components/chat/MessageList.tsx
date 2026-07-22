import React, { useLayoutEffect, useRef, useState, useCallback } from 'react';
import { MessageBubble } from './MessageBubble';

import type { Message } from '../../types/chat';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  streamingContent: string;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  streamingContent,
}) => {
  const listRef = useRef<HTMLDivElement>(null);
  const prevMsgCountRef = useRef(messages.length);
  const userScrolledUpRef = useRef(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const isNearBottom = useCallback(() => {
    const el = listRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  const scrollToBottom = useCallback((smooth: boolean) => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'instant' as ScrollBehavior });
  }, []);

  // 新消息到达时的滚动
  useLayoutEffect(() => {
    const msgCount = messages.length;
    const isNewMessage = msgCount > prevMsgCountRef.current;
    prevMsgCountRef.current = msgCount;

    // 新消息（用户发送或 AI 开始回复）→ 强制滚到底部
    if (isNewMessage) {
      userScrolledUpRef.current = false;
      scrollToBottom(true);
    }
  }, [messages, scrollToBottom]);

  // streaming 时的滚动 — 用 instant 避免动画冲突
  useLayoutEffect(() => {
    if (isLoading && streamingContent && !userScrolledUpRef.current) {
      scrollToBottom(false);
    }
  }, [streamingContent, isLoading, scrollToBottom]);

  // 监听用户手动滚动
  useLayoutEffect(() => {
    const el = listRef.current;
    if (!el) return;

    const handleScroll = () => {
      userScrolledUpRef.current = !isNearBottom();
      setShowScrollBtn(!isNearBottom());
    };

    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  }, [isNearBottom]);

  return (
    <div className="message-list" ref={listRef}>
      {messages.length === 0 && !isLoading && (
        <div className="empty-chat">
          <h2>🤖 智能客服</h2>
          <p>上传知识库文档后，开始提问吧</p>
        </div>
      )}
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1;
        const isAssistant = msg.role === 'assistant';
        // 流式进行中：最后一条已有 assistant 消息时，inline 展示流式内容
        const showInlineStreaming = isLast && isLoading && isAssistant;
        return (
          <MessageBubble
            key={msg.id}
            message={msg}
            isStreaming={showInlineStreaming}
            streamingContent={showInlineStreaming ? streamingContent : undefined}
          />
        );
      })}

      {/* 流式进行中且还没有 assistant 消息时，显示「幽灵气泡」承载流式内容 */}
      {isLoading && (messages.length === 0 || messages[messages.length - 1].role !== 'assistant') && (
        <MessageBubble
          message={{ id: 'streaming-ghost', role: 'assistant', content: '' }}
          isStreaming={true}
          streamingContent={streamingContent}
        />
      )}
      {showScrollBtn && (
        <button
          className="scroll-to-bottom"
          onClick={() => {
            userScrolledUpRef.current = false;
            scrollToBottom(true);
          }}
        >
          ↓ 回到底部
        </button>
      )}
    </div>
  );
};
