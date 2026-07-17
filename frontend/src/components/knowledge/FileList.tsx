import React from 'react';
import { FileListItem } from './FileListItem';

interface FileListProps {
  files: string[];
  onDelete: (filename: string) => void;
}

export const FileList: React.FC<FileListProps> = ({ files, onDelete }) => {
  if (files.length === 0) {
    return <p className="empty-hint">暂无文件</p>;
  }

  return (
    <div className="file-list">
      {files.map((f) => (
        <FileListItem key={f} filename={f} onDelete={onDelete} />
      ))}
    </div>
  );
};
