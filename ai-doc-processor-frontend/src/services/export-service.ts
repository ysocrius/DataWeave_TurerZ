import apiClient from './api-client';
import type { ExportOptions, ExportJob } from '@/types';

class ExportService {
  async createExport(sessionId: string, options: ExportOptions): Promise<ExportJob> {
    const response = await apiClient.post<ExportJob>(`/processing/${sessionId}/export`, options);
    return response.data;
  }

  async getExportStatus(exportId: string): Promise<ExportJob> {
    const response = await apiClient.get<ExportJob>(`/exports/${exportId}`);
    return response.data;
  }

  async downloadExport(exportId: string): Promise<Blob> {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'}/exports/${exportId}/download`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  async deleteExport(exportId: string): Promise<void> {
    await apiClient.delete(`/exports/${exportId}`);
  }

  async getExports(): Promise<ExportJob[]> {
    const response = await apiClient.get<ExportJob[]>('/exports');
    return response.data;
  }

  downloadBlob(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }
}

const exportService = new ExportService();
export default exportService;