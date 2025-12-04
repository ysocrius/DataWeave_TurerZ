import apiClient from './api-client';
import type { UploadedFile } from '@/types';

class FileService {
  async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<UploadedFile> {
    const response = await apiClient.upload<UploadedFile>('/files/upload', file, onProgress);
    return response.data;
  }

  async getFile(fileId: string): Promise<UploadedFile> {
    const response = await apiClient.get<UploadedFile>(`/files/${fileId}`);
    return response.data;
  }

  async deleteFile(fileId: string): Promise<void> {
    await apiClient.delete(`/files/${fileId}`);
  }

  async getFiles(): Promise<UploadedFile[]> {
    const response = await apiClient.get<UploadedFile[]>('/files');
    return response.data;
  }

  validateFile(file: File): { isValid: boolean; error?: string } {
    // File type validation
    const allowedTypes = ['application/pdf'];
    if (!allowedTypes.includes(file.type)) {
      return {
        isValid: false,
        error: 'Only PDF files are allowed',
      };
    }

    // File size validation (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB in bytes
    if (file.size > maxSize) {
      return {
        isValid: false,
        error: 'File size must be less than 50MB',
      };
    }

    return { isValid: true };
  }
}

const fileService = new FileService();
export default fileService;