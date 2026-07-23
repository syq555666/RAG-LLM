import { API_BASE_URL } from '../utils/constants';
import type { StreamEvent, TokenData, ToolStartData, ToolEndData, DoneData, ErrorData } from '../types/chat';

export async function* streamChat(
  sessionId: string,
  message: string,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    let currentEvent = '';
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith('data: ')) {
        const rawData = JSON.parse(line.slice(6));
        const eventType = currentEvent as StreamEvent['event'];

        let data: TokenData | ToolStartData | ToolEndData | DoneData | ErrorData;
        switch (eventType) {
          case 'token': data = rawData as TokenData; break;
          case 'tool_start': data = rawData as ToolStartData; break;
          case 'tool_end': data = rawData as ToolEndData; break;
          case 'done': data = rawData as DoneData; break;
          case 'error': data = rawData as ErrorData; break;
          default: data = rawData;
        }

        yield { event: eventType, data };
      }
    }
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
