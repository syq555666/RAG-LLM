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

  return (
    <div className="chat-container">
      <MessageList
        messages={messages}
        isLoading={isLoading}
        streamingContent={streamingContent}
      />
      <SuggestionChips suggestions={suggestions} onSelect={sendMessage} />
      <ChatInput onSend={sendMessage} onStop={stopGeneration} isLoading={isLoading} />
    </div>
  );
};
