import React, { useCallback, useRef, useState, type DragEvent, type ChangeEvent } from 'react';
import { ALLOWED_FILE_TYPES } from '../../utils/constants';

interface FileUploaderProps {
  onUpload: (files: File[]) => void;
  disabled?: boolean;
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onUpload, disabled }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files) return;
      const validFiles: File[] = [];
      for (const file of Array.from(files)) {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase();
        if (ALLOWED_FILE_TYPES.includes(ext)) {
          validFiles.push(file);
        }
      }
      if (validFiles.length > 0) {
        onUpload(validFiles);
      }
    },
    [onUpload]
  );

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (!disabled) handleFiles(e.dataTransfer.files);
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div
      className={`file-uploader ${isDragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".txt,.md,.csv,.json"
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <div className="upload-icon">📤</div>
      <p>{isDragOver ? '松手上传' : '点击或拖拽文件到此处'}</p>
      <span className="upload-hint">支持 txt, md, csv, json</span>
    </div>
  );
};
