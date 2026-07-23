import { useCallback, useEffect, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { listSessions, createSession, getHistory, deleteSession as apiDeleteSession } from '../api/sessions';
import toast from 'react-hot-toast';
import type { SessionInfo } from '../types/session';

const SESSION_KEY = 'rag_chat_session_id';

export function useSession() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const sessionId = useChatStore((s) => s.sessionId);
  const setSessionId = useChatStore((s) => s.setSessionId);
  const clearSessionId = useChatStore((s) => s.clearSessionId);
  const loadHistory = useChatStore((s) => s.loadHistory);
  const clearMessages = useChatStore((s) => s.clearMessages);

  const initSession = useCallback(async () => {
    // 尝试从 localStorage 加载已有 session
    let sid = localStorage.getItem(SESSION_KEY);

    if (sid) {
      try {
        const history = await getHistory(sid);
        setSessionId(sid);
        loadHistory(history.messages);
      } catch {
        // session 不存在了，创建新的
        sid = null;
      }
    }

    if (!sid) {
      const res = await createSession();
      sid = res.session_id;
      localStorage.setItem(SESSION_KEY, sid);
      setSessionId(sid);
    }

    // 加载会话列表
    try {
      const data = await listSessions();
      setSessions(data.sessions);
    } catch {
      // 静默失败
    }
  }, [setSessionId, loadHistory]);

  const newChat = useCallback(async () => {
    const res = await createSession();
    localStorage.setItem(SESSION_KEY, res.session_id);
    setSessionId(res.session_id);
    clearMessages();
    try {
      const data = await listSessions();
      setSessions(data.sessions);
    } catch { /* */ }
  }, [setSessionId, clearMessages]);

  const switchSession = useCallback(
    async (targetId: string) => {
      localStorage.setItem(SESSION_KEY, targetId);
      setSessionId(targetId);
      try {
        const history = await getHistory(targetId);
        loadHistory(history.messages);
      } catch {
        clearMessages();
      }
    },
    [setSessionId, loadHistory, clearMessages]
  );

  const deleteSession = useCallback(
    async (targetId: string) => {
      try {
        await apiDeleteSession(targetId);
        toast.success('会话已删除');
        setSessions((prev) => prev.filter((s) => s.id !== targetId));
        if (sessionId === targetId) {
          // 删除当前会话 → 清空状态，不自动创建（允许全部删除）
          localStorage.removeItem(SESSION_KEY);
          clearSessionId();
          clearMessages();
        }
      } catch {
        toast.error('删除失败');
      }
    },
    [sessionId, clearSessionId, clearMessages]
  );

  useEffect(() => {
    initSession();
  }, [initSession]);

  return {
    sessions,
    sessionId,
    newChat,
    switchSession,
    deleteSession,
    refreshSessions: () =>
      listSessions().then((d) => setSessions(d.sessions)).catch(() => {}),
  };
}
