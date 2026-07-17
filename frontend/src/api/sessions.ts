import { apiPost } from './client';
import type { HistoryData, SessionInfo } from '../types/session';

export async function listSessions(): Promise<{ sessions: SessionInfo[] }> {
  const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/sessions`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function createSession(): Promise<{ session_id: string }> {
  return apiPost('/api/sessions');
}

export async function getHistory(sessionId: string): Promise<HistoryData> {
  const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/sessions/${sessionId}/history`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
}
