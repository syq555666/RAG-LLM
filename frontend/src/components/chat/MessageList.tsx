import React, { useEffect, useRef } from 'react';
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
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  return (
    <div className="message-list">
      {messages.length === 0 && !isLoading && (
        <div className="empty-chat">
          <h2>🤖 智能客服</h2>
          <p>上传知识库文档后，开始提问吧</p>
        </div>
      )}
      {messages.map((msg, i) => {
        const isLast = i === messages.length - 1;
        const isAssistant = msg.role === 'assistant';
        return (
          <MessageBubble
            key={msg.id}
            message={msg}
            isStreaming={isLast && isLoading && isAssistant}
            streamingContent={isLast && isLoading && isAssistant ? streamingContent : undefined}
          />
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
};
