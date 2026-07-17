import { apiGet, apiDelete } from './client';
import type { FileListData, FileUploadResult } from '../types/knowledge';
import { API_BASE_URL } from '../utils/constants';

export async function listFiles(): Promise<FileListData> {
  return apiGet<FileListData>('/api/knowledge/files');
}

export async function uploadFiles(files: File[]): Promise<FileUploadResult[]> {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));

  const res = await fetch(`${API_BASE_URL}/api/knowledge/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.results;
}

export async function deleteFile(filename: string): Promise<void> {
  await apiDelete(`/api/knowledge/files/${encodeURIComponent(filename)}`);
}
