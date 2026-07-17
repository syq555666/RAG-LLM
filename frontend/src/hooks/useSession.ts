import { useCallback, useEffect, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { listSessions, createSession, getHistory, deleteSession as apiDeleteSession } from '../api/sessions';
import toast from 'react-hot-toast';
import type { SessionInfo } from '../types/session';

const SESSION_KEY = 'rag_chat_session_id';

export function useSession() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const store = useChatStore();

  const initSession = useCallback(async () => {
    // 尝试从 localStorage 加载已有 session
    let sessionId = localStorage.getItem(SESSION_KEY);

    if (sessionId) {
      try {
        const history = await getHistory(sessionId);
        store.setSessionId(sessionId);
        store.loadHistory(history.messages);
      } catch {
        // session 不存在了，创建新的
        sessionId = null;
      }
    }

    if (!sessionId) {
      const res = await createSession();
      sessionId = res.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
      store.setSessionId(sessionId);
    }

    // 加载会话列表
    try {
      const data = await listSessions();
      setSessions(data.sessions);
    } catch {
      // 静默失败
    }
  }, [store]);

  const newChat = useCallback(async () => {
    const res = await createSession();
    localStorage.setItem(SESSION_KEY, res.session_id);
    store.setSessionId(res.session_id);
    store.clearMessages();
    try {
      const data = await listSessions();
      setSessions(data.sessions);
    } catch { /* */ }
  }, [store]);

  const switchSession = useCallback(
    async (sessionId: string) => {
      localStorage.setItem(SESSION_KEY, sessionId);
      store.setSessionId(sessionId);
      try {
        const history = await getHistory(sessionId);
        store.loadHistory(history.messages);
      } catch {
        store.clearMessages();
      }
    },
    [store]
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        await apiDeleteSession(sessionId);
        toast.success('会话已删除');
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        if (store.sessionId === sessionId) {
          await newChat();
        }
      } catch {
        toast.error('删除失败');
      }
    },
    [store, newChat]
  );

  useEffect(() => {
    initSession();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    sessions,
    sessionId: store.sessionId,
    newChat,
    switchSession,
    deleteSession,
    refreshSessions: () =>
      listSessions().then((d) => setSessions(d.sessions)).catch(() => {}),
  };
}
