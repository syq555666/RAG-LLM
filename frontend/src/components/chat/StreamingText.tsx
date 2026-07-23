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
    if (content.length > lastLenRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        setDisplayed(content);
        lastLenRef.current = content.length;
      });
    }
  }, [content]);

  useEffect(() => {
    if (!isStreaming) {
      setDisplayed(content);
      lastLenRef.current = content.length;
    }
  }, [isStreaming, content]);

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
