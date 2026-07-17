import React from 'react';

interface FileListItemProps {
  filename: string;
  onDelete: (filename: string) => void;
}

export const FileListItem: React.FC<FileListItemProps> = ({ filename, onDelete }) => {
  const [confirming, setConfirming] = React.useState(false);

  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const icon = { txt: '📄', md: '📝', csv: '📊', json: '📋' }[ext] || '📄';

  return (
    <div className="file-list-item">
      <span className="file-icon">{icon}</span>
      <span className="file-name" title={filename}>{filename}</span>
      {confirming ? (
        <span className="delete-confirm">
          <button onClick={() => { onDelete(filename); setConfirming(false); }}>确认</button>
          <button onClick={() => setConfirming(false)}>取消</button>
        </span>
      ) : (
        <button className="btn-delete" onClick={() => setConfirming(true)} title="删除">🗑️</button>
      )}
    </div>
  );
};
