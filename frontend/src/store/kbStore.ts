import { create } from 'zustand';
import { listFiles } from '../api/knowledge';

interface KBState {
  files: string[];
  fileCount: number;
  isUploading: boolean;

  setFiles: (files: string[], count: number) => void;
  setUploading: (val: boolean) => void;
  refresh: () => Promise<void>;
}

export const useKBStore = create<KBState>((set) => ({
  files: [],
  fileCount: 0,
  isUploading: false,

  setFiles: (files, count) => set({ files, fileCount: count }),
  setUploading: (val) => set({ isUploading: val }),

  refresh: async () => {
    try {
      const data = await listFiles();
      set({ files: data.files, fileCount: data.count });
    } catch {
      // 静默失败
    }
  },
}));
