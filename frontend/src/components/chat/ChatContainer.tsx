import React from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { SuggestionChips } from './SuggestionChips';
import { useChatStore } from '../../store/chatStore';
import { useChatStream } from '../../hooks/useChatStream';
import { useSession } from '../../hooks/useSession';

export const ChatContainer: React.FC = () => {
  const { sendMessage, stopGeneration, isLoading } = useChatStream();
  const messages = useChatStore((s) => s.messages);
  const streamingContent = useChatStore((s) => s.streamingContent);
  const suggestions = useChatStore((s) => s.suggestions);
  const sessionId = useChatStore((s) => s.sessionId);
  const { newChat } = useSession();

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
          <p>开始一个新的对话，探索知识库中的内容</p>
          <button className="btn-new-chat" onClick={newChat} style={{ marginTop: 12, fontSize: 14, padding: '8px 20px' }}>
            + 新会话
          </button>
        </div>
      )}
    </div>
  );
};
