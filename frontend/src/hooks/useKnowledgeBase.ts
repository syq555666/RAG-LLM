import { useCallback, useEffect } from 'react';
import { useKBStore } from '../store/kbStore';
import { uploadFiles as apiUploadFiles, deleteFile as apiDeleteFile } from '../api/knowledge';
import toast from 'react-hot-toast';
import type { FileUploadResult } from '../types/knowledge';

export function useKnowledgeBase() {
  const store = useKBStore();

  useEffect(() => {
    store.refresh();
  }, [store]);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      store.setUploading(true);
      try {
        const results: FileUploadResult[] = await apiUploadFiles(files);
        const success = results.filter((r) => r.status === 'success').length;
        const errors = results.filter((r) => r.status === 'error').length;

        if (success > 0) toast.success(`成功上传 ${success} 个文件`);
        if (errors > 0) toast.error(`${errors} 个文件上传失败`);

        results.forEach((r) => {
          if (r.status === 'success' && r.message) {
            console.log(`${r.filename}: ${r.message}`);
          }
        });

        await store.refresh();
      } catch (err) {
        toast.error('文件上传失败');
        console.error(err);
      } finally {
        store.setUploading(false);
      }
    },
    [store]
  );

  const deleteFile = useCallback(
    async (filename: string) => {
      try {
        await apiDeleteFile(filename);
        toast.success(`已删除 ${filename}`);
        await store.refresh();
      } catch {
        toast.error('删除失败');
      }
    },
    [store]
  );

  return {
    files: store.files,
    fileCount: store.fileCount,
    isUploading: store.isUploading,
    uploadFiles,
    deleteFile,
    refresh: store.refresh,
  };
}
