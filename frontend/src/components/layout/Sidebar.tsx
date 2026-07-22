import React from 'react';
import { FileUploader } from '../knowledge/FileUploader';
import { FileList } from '../knowledge/FileList';
import { Modal } from '../common/Modal';
import { useKnowledgeBase } from '../../hooks/useKnowledgeBase';
import { useSession } from '../../hooks/useSession';
import { Spinner } from '../common/Spinner';
import { formatTime } from '../../utils/format';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const { files, fileCount, isUploading, uploadFiles, deleteFile, refresh } = useKnowledgeBase();
  const { sessions, sessionId, newChat, switchSession, deleteSession } = useSession();
  const [deleteTarget, setDeleteTarget] = React.useState<string | null>(null);

  const handleSwitchSession = (id: string) => {
    switchSession(id);
    onClose(); // 移动端切换后关闭侧边栏
  };

  const handleNewChat = () => {
    newChat();
    onClose();
  };

  const handleDeleteSession = (id: string) => {
    setDeleteTarget(id);
  };

  const confirmDeleteSession = () => {
    if (deleteTarget) {
      deleteSession(deleteTarget);
      setDeleteTarget(null);
    }
  };

  return (
    <>
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>📚 知识库管理</h2>
          <div className="kb-status">
            📊 {fileCount} 个文件
            <button className="btn-refresh" onClick={refresh} title="刷新">🔄</button>
          </div>
        </div>

        <div className="sidebar-section">
          <FileUploader onUpload={uploadFiles} disabled={isUploading} />
          {isUploading && (
            <div className="uploading-indicator">
              <Spinner size={14} /> 上传中...
            </div>
          )}
        </div>

        <div className="sidebar-section">
          <h3>🗑️ 文件管理</h3>
          <FileList files={files} onDelete={deleteFile} />
          <span className="section-hint">💡 支持 txt, md, csv, json</span>
        </div>

        <div className="sidebar-divider" />

        <div className="sidebar-section">
          <div className="session-header">
            <h3>💬 会话历史</h3>
            <button className="btn-new-chat" onClick={handleNewChat}>+ 新会话</button>
          </div>
          <div className="session-list">
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`session-item ${s.id === sessionId ? 'active' : ''}`}
                onClick={() => handleSwitchSession(s.id)}
              >
                <span className="session-preview">{s.preview || '(空)'}</span>
                <span className="session-time">{formatTime(s.updated_at)}</span>
                <button
                  className="btn-session-delete"
                  onClick={(e) => { e.stopPropagation(); handleDeleteSession(s.id); }}
                  title="删除"
                >
                  🗑️
                </button>
              </div>
            ))}
            {sessions.length === 0 && (
              <p className="empty-hint">暂无会话</p>
            )}
          </div>
        </div>
      </aside>

      <Modal
        isOpen={deleteTarget !== null}
        title="删除会话"
        message="确定要删除这个会话吗？对应的对话历史也将被清除。"
        onConfirm={confirmDeleteSession}
        onCancel={() => setDeleteTarget(null)}
      />
    </>
  );
};
