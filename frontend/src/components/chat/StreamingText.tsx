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

  // 流式进行中 → 纯文本渲染，避免逐 token 触发 Markdown 解析
  if (isStreaming) {
    return (
      <div className="streaming-text">
        <span className="streaming-text-raw">{displayed}</span>
        <span className="cursor-blink">▌</span>
      </div>
    );
  }

  // 流式结束 → Markdown 渲染
  return (
    <div className="streaming-text">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayed}</ReactMarkdown>
    </div>
  );
};
