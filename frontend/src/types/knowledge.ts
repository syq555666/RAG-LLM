export interface FileUploadResult {
  filename: string;
  status: 'success' | 'error';
  chunks_added: number;
  chunks_skipped: number;
  message: string;
}

export interface FileListData {
  files: string[];
  count: number;
}
