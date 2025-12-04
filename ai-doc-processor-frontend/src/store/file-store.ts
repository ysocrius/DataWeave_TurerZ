import { create } from 'zustand';
import type { UploadedFile } from '@/types';
import { fileService } from '@/services';

interface FileState {
  files: UploadedFile[];
  currentFile: UploadedFile | null;
  uploadProgress: Record<string, number>;
  isLoading: boolean;
  error: string | null;
}

interface FileActions {
  uploadFile: (file: File) => Promise<UploadedFile>;
  getFiles: () => Promise<void>;
  getFile: (fileId: string) => Promise<void>;
  deleteFile: (fileId: string) => Promise<void>;
  setCurrentFile: (file: UploadedFile | null) => void;
  updateUploadProgress: (fileId: string, progress: number) => void;
  clearError: () => void;
}

type FileStore = FileState & FileActions;

export const useFileStore = create<FileStore>((set, get) => ({
  // State
  files: [],
  currentFile: null,
  uploadProgress: {},
  isLoading: false,
  error: null,

  // Actions
  uploadFile: async (file: File) => {
    const validation = fileService.validateFile(file);
    if (!validation.isValid) {
      set({ error: validation.error });
      throw new Error(validation.error);
    }

    set({ isLoading: true, error: null });
    
    try {
      const uploadedFile = await fileService.uploadFile(file, (progress) => {
        get().updateUploadProgress(file.name, progress);
      });

      set(state => ({
        files: [...state.files, uploadedFile],
        isLoading: false,
        uploadProgress: { ...state.uploadProgress, [file.name]: 100 }
      }));

      return uploadedFile;
    } catch (error: any) {
      set({ 
        error: error.message || 'Upload failed', 
        isLoading: false 
      });
      throw error;
    }
  },

  getFiles: async () => {
    set({ isLoading: true, error: null });
    try {
      const files = await fileService.getFiles();
      set({ files, isLoading: false });
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to fetch files', 
        isLoading: false 
      });
    }
  },

  getFile: async (fileId: string) => {
    set({ isLoading: true, error: null });
    try {
      const file = await fileService.getFile(fileId);
      set({ currentFile: file, isLoading: false });
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to fetch file', 
        isLoading: false 
      });
    }
  },

  deleteFile: async (fileId: string) => {
    set({ isLoading: true, error: null });
    try {
      await fileService.deleteFile(fileId);
      set(state => ({
        files: state.files.filter(file => file.id !== fileId),
        currentFile: state.currentFile?.id === fileId ? null : state.currentFile,
        isLoading: false
      }));
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to delete file', 
        isLoading: false 
      });
    }
  },

  setCurrentFile: (file: UploadedFile | null) => {
    set({ currentFile: file });
  },

  updateUploadProgress: (fileId: string, progress: number) => {
    set(state => ({
      uploadProgress: {
        ...state.uploadProgress,
        [fileId]: progress
      }
    }));
  },

  clearError: () => set({ error: null }),
}));