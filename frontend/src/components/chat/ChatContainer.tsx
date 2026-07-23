import React from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { SuggestionChips } from './SuggestionChips';
import { useChatStore } from '../../store/chatStore';
import { useChatStream } from '../../hooks/useChatStream';

export const ChatContainer: React.FC = () => {
  const { sendMessage, stopGeneration, isLoading } = useChatStream();
  const messages = useChatStore((s) => s.messages);
  const streamingContent = useChatStore((s) => s.streamingContent);
  const suggestions = useChatStore((s) => s.suggestions);
  const sessionId = useChatStore((s) => s.sessionId);

  return (
    <div className="chat-container">
      {sessionId ? (
        <>
          <MessageList
            messages={messages}
            isLoading={isLoading}
            streamingContent={streamingContent}
          />
          <SuggestionChips suggestions={suggestions} onSelect={sendMessage} />
          <ChatInput onSend={sendMessage} onStop={stopGeneration} isLoading={isLoading} />
        </>
      ) : (
        <div className="empty-chat">
          <h2>🤖 智能客服</h2>
          <p>点击左侧「+ 新会话」开始对话，或选择已有会话</p>
        </div>
      )}
    </div>
  );
};
