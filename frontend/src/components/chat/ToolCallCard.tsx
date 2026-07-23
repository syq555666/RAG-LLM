import React, { useState } from 'react';
import { Spinner } from '../common/Spinner';
import type { ToolCallRecord } from '../../types/chat';

const TOOL_LABELS: Record<string, string> = {
  get_current_time: '获取当前时间',
  web_search: '网络搜索',
  search_knowledge_base: '搜索知识库',
};

interface ToolCallCardProps {
  toolCall: ToolCallRecord;
}

export const ToolCallCard: React.FC<ToolCallCardProps> = ({ toolCall }) => {
  const [expanded, setExpanded] = useState(false);
  const label = TOOL_LABELS[toolCall.toolName] || toolCall.toolName;

  return (
    <div className={`tool-call-card ${toolCall.status}`}>
      <div className="tool-call-header" onClick={() => setExpanded(!expanded)}>
        {toolCall.status === 'running' ? (
          <Spinner size={14} />
        ) : toolCall.status === 'done' ? (
          <span className="tool-icon">✓</span>
        ) : (
          <span className="tool-icon">✗</span>
        )}
        <span className="tool-label">
          {toolCall.status === 'running' ? `正在${label}...` : `${label}完成`}
        </span>
        {toolCall.result && (
          <button className="tool-expand-btn">{expanded ? '收起' : '详情'}</button>
        )}
      </div>
      {expanded && toolCall.result && (
        <div className="tool-call-result">
          <pre>{toolCall.result}</pre>
        </div>
      )}
    </div>
  );
};
