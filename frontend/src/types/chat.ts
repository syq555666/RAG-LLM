export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: ToolCallRecord[];
}

export interface ToolCallRecord {
  toolName: string;
  args?: Record<string, unknown>;
  result?: string;
  status: 'running' | 'done' | 'error';
}

export interface StreamEvent {
  event: 'tool_start' | 'tool_end' | 'token' | 'done' | 'error';
  data: TokenData | ToolStartData | ToolEndData | DoneData | ErrorData;
}

export interface TokenData {
  content: string;
}

export interface ToolStartData {
  tool_name: string;
  args: Record<string, unknown>;
}

export interface ToolEndData {
  tool_name: string;
  result: string;
}

export interface DoneData {
  full_response: string;
}

export interface ErrorData {
  error: string;
}
