import React, { useState, useRef, useEffect, type KeyboardEvent } from 'react';
import { useChatStore } from '../../store/chatStore';
import { useKBStore } from '../../store/kbStore';

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isLoading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, onStop, isLoading }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileCount = useKBStore((s) => s.fileCount);
  const sessionId = useChatStore((s) => s.sessionId);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  // 发送后自动聚焦
  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isLoading]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getPlaceholder = () => {
    if (!sessionId) return '请先创建或选择一个会话...';
    if (fileCount === 0) return '💡 请先在左侧上传知识库文档...';
    return '💬 输入你的问题，我会从知识库中寻找答案...';
  };

  return (
    <div className="chat-input-container">
      <textarea
        ref={textareaRef}
        className="chat-input"
        placeholder={getPlaceholder()}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        disabled={isLoading}
      />
      <div className="chat-input-actions">
        <button
          className="btn-send"
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
        >
          发送
        </button>
        {isLoading && (
          <button className="btn-stop" onClick={onStop}>
            ⏹ 停止
          </button>
        )}
      </div>
    </div>
  );
};
