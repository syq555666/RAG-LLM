import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface StreamingTextProps {
  content: string;
  isStreaming: boolean;
}

export const StreamingText: React.FC<StreamingTextProps> = ({ content, isStreaming }) => {
  const [displayed, setDisplayed] = useState('');
  const rafRef = useRef<number>(0);
  const lastLenRef = useRef(0);

  useEffect(() => {
    // 流式：仅内容增长时通过 rAF 更新，避免高频渲染
    // 非流式：直接展示完整内容
    if (!isStreaming || content.length <= lastLenRef.current) {
      setDisplayed(content);
      lastLenRef.current = content.length;
      return;
    }
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      setDisplayed(content);
      lastLenRef.current = content.length;
    });
  }, [content, isStreaming]);

  useEffect(() => {
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <div className="streaming-text">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayed}</ReactMarkdown>
      {isStreaming && <span className="cursor-blink">▌</span>}
    </div>
  );
};
