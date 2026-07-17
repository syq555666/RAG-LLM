export interface SessionInfo {
  id: string;
  preview: string | null;
  updated_at: string;
}

export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface HistoryData {
  messages: HistoryMessage[];
  summary: string | null;
}
