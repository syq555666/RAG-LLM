import { apiGet, apiPost, apiDelete } from './client';
import type { HistoryData, SessionInfo } from '../types/session';

export async function listSessions(): Promise<{ sessions: SessionInfo[] }> {
  return apiGet<{ sessions: SessionInfo[] }>('/api/sessions');
}

export async function createSession(): Promise<{ session_id: string }> {
  return apiPost('/api/sessions');
}

export async function getHistory(sessionId: string): Promise<HistoryData> {
  return apiGet<HistoryData>(`/api/sessions/${sessionId}/history`);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiDelete(`/api/sessions/${sessionId}`);
}
