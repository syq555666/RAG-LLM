import { useCallback, useEffect } from 'react';
import { useKBStore } from '../store/kbStore';
import { uploadFiles as apiUploadFiles, deleteFile as apiDeleteFile } from '../api/knowledge';
import toast from 'react-hot-toast';
import type { FileUploadResult } from '../types/knowledge';

export function useKnowledgeBase() {
  const files = useKBStore((s) => s.files);
  const fileCount = useKBStore((s) => s.fileCount);
  const isUploading = useKBStore((s) => s.isUploading);
  const setUploading = useKBStore((s) => s.setUploading);
  const refresh = useKBStore((s) => s.refresh);

  useEffect(() => {
    refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const uploadFiles = useCallback(
    async (fileList: File[]) => {
      setUploading(true);
      try {
        const results: FileUploadResult[] = await apiUploadFiles(fileList);
        const success = results.filter((r) => r.status === 'success').length;
        const errors = results.filter((r) => r.status === 'error').length;

        if (success > 0) toast.success(`成功上传 ${success} 个文件`);
        if (errors > 0) toast.error(`${errors} 个文件上传失败`);

        results.forEach((r) => {
          if (r.status === 'success' && r.message) {
            console.log(`${r.filename}: ${r.message}`);
          }
        });

        await refresh();
      } catch (err) {
        toast.error('文件上传失败');
        console.error(err);
      } finally {
        setUploading(false);
      }
    },
    [setUploading, refresh]
  );

  const deleteFile = useCallback(
    async (filename: string) => {
      try {
        await apiDeleteFile(filename);
        toast.success(`已删除 ${filename}`);
        await refresh();
      } catch {
        toast.error('删除失败');
      }
    },
    [refresh]
  );

  return { files, fileCount, isUploading, uploadFiles, deleteFile, refresh };
}
