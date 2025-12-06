// Processing Service - Handles PDF upload and AI processing

export interface ProcessingResult {
  entries: Array<{
    '#'?: number;
    Key: string;
    Value: string;
    Comments?: string;
  }>;
  global_notes?: string;
  processing_time?: number;
  session_id?: string;
}

export interface ChunkInfo {
  chunk_id: number;
  page_range?: string;
  pages?: number[];
  section_type?: string;
  header?: string;
  entries_count: number;
  raw_entries_count?: number;
  duplicates_removed?: number;
  processing_time: number;
  priority?: number;
  content_length?: number;
  total_length_with_context?: number;
  status: 'success' | 'error';
  error?: string;
}

export interface ChunkedProcessingResult extends ProcessingResult {
  chunks: ChunkInfo[];
  chunk_results: Array<{
    chunk_id: number;
    entries: Array<any>;
  }>;
  total_pages: number;
  total_chunks: number;
  processing_time: number;
  chunk_size: number;
}

export interface ProcessingJob {
  id: string;
  status: 'pending' | 'processing' | 'complete' | 'error';
  fileName: string;
  result?: ProcessingResult;
  error?: string;
}

// SSE Event Types
export interface ChunkProgressEvent {
  event: 'start' | 'chunk_start' | 'chunk_progress' | 'chunk_complete' | 'complete' | 'error' |
  'analysis_start' | 'analysis_complete' | 'merging_start' | 'intelligent_complete' | 'character_complete' |
  'enhanced_complete' | 'semantic_complete' | 'context_initialized' | 'batch_processing_start' | 
  'batch_processing_complete' | 'dedup_progress' | 'dedup_complete' | 'position_tracking_start' | 'position_tracking_complete';
  chunk?: number;
  total_chunks?: number;
  page_range?: string;
  section_type?: string;
  header?: string;
  char_range?: string;
  status?: string;
  entries_count?: number;
  processing_time?: number;
  error?: string;
  message?: string;
  // Analysis phase
  sections_found?: number;
  strategy?: string;
  chunk_size?: number;
  overlap?: number;
  // Complete event includes full result
  entries?: Array<any>;
  global_notes?: string;
  chunks?: ChunkInfo[];
  chunk_results?: Array<{ chunk_id: number; entries: Array<any> }>;
  total_rows?: number;
  total_pages?: number;
  processing_strategy?: string;
  sections_analyzed?: number;
  merge_strategy?: string;
  session_id?: string;
}

export type ChunkProgressCallback = (event: ChunkProgressEvent) => void;

class ProcessingService {
  private readonly API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8001';

  /**
   * Check if backend is available
   */
  async checkBackendHealth(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const response = await fetch(`${this.API_BASE_URL}/`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      return false;
    }
  }



  /**
   * Process text input using enhanced semantic chunking with batch parallel processing and context memory
   */
  async processText(
    text: string,
    onProgress: ChunkProgressCallback
  ): Promise<ChunkedProcessingResult> {
    // Validate text
    if (!text.trim()) {
      throw new Error('Text input is empty. Please provide some text to process.');
    }

    if (text.length > 50000) {
      throw new Error('Text is too long. Please limit to 50,000 characters.');
    }

    return new Promise((resolve, reject) => {
      fetch(`${this.API_BASE_URL}/api/process-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
      })
        .then(async (response) => {
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('Failed to read response stream');
          }

          let buffer = '';
          let finalResult: ChunkedProcessingResult | null = null;

          const processStream = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;

                buffer += new TextDecoder().decode(value);
                
                // SSE events are separated by double newlines
                const events = buffer.split('\n\n');
                buffer = events.pop() || '';

                for (const event of events) {
                  const lines = event.split('\n');
                  for (const line of lines) {
                    if (line.startsWith('data: ')) {
                      try {
                        const eventData = JSON.parse(line.slice(6));
                        
                        if (eventData.event === 'enhanced_complete' || eventData.event === 'semantic_complete') {
                          finalResult = eventData as ChunkedProcessingResult;
                        }
                        
                        onProgress(eventData);
                      } catch (parseError) {
                        console.warn('Failed to parse SSE data:', line);
                      }
                    }
                  }
                }
              }

              if (finalResult) {
                resolve(finalResult);
              } else {
                throw new Error('Processing completed but no final result received');
              }
            } catch (streamError) {
              reject(streamError);
            }
          };

          processStream();
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  /**
   * Process file from file path using enhanced semantic chunking with SSE streaming
   */
  async processFilePath(
    filePath: string,
    onProgress: ChunkProgressCallback
  ): Promise<ChunkedProcessingResult> {
    // Validate file path
    if (!filePath.trim()) {
      throw new Error('File path is empty. Please provide a valid file path.');
    }

    return new Promise((resolve, reject) => {
      fetch(`${this.API_BASE_URL}/api/process-filepath`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ file_path: filePath }),
      })
        .then(async (response) => {
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('Failed to read response stream');
          }

          let buffer = '';
          let finalResult: ChunkedProcessingResult | null = null;

          const processStream = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;

                buffer += new TextDecoder().decode(value);
                
                // SSE events are separated by double newlines
                const events = buffer.split('\n\n');
                buffer = events.pop() || '';

                for (const event of events) {
                  const lines = event.split('\n');
                  for (const line of lines) {
                    if (line.startsWith('data: ')) {
                      try {
                        const eventData = JSON.parse(line.slice(6));
                        
                        if (eventData.event === 'enhanced_complete') {
                          finalResult = eventData as ChunkedProcessingResult;
                        }
                        
                        onProgress(eventData);
                      } catch (parseError) {
                        console.warn('Failed to parse SSE data:', line);
                      }
                    }
                  }
                }
              }

              if (finalResult) {
                resolve(finalResult);
              } else {
                throw new Error('Processing completed but no final result received');
              }
            } catch (streamError) {
              reject(streamError);
            }
          };

          processStream();
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  /**
   * Download Excel file
   */
  async downloadExcel(data: ProcessingResult, filename: string = 'AI_Extracted_Data.xlsx'): Promise<void> {
    try {
      const response = await fetch(`${this.API_BASE_URL}/api/download-excel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error('Failed to generate Excel file');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error: any) {
      throw new Error(error.message || 'Failed to download Excel file');
    }
  }





  /**
   * Process a PDF file using enhanced semantic chunking with batch parallel processing and context memory
   */
  async processPDF(
    file: File,
    onProgress: ChunkProgressCallback
  ): Promise<ChunkedProcessingResult> {
    // Validate file
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      throw new Error('Invalid file type. Please upload a PDF file.');
    }

    if (file.size > 10 * 1024 * 1024) {
      throw new Error('File size exceeds 10MB limit. Please upload a smaller file.');
    }

    // Check backend availability
    const isBackendAvailable = await this.checkBackendHealth();
    if (!isBackendAvailable) {
      throw new Error('Backend server is not responding. Please ensure the Python backend is running.');
    }

    const formData = new FormData();
    formData.append('file', file);

    return new Promise((resolve, reject) => {
      fetch(`${this.API_BASE_URL}/api/process`, {
        method: 'POST',
        body: formData,
      })
        .then(async (response) => {
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('Failed to get response reader');
          }

          const decoder = new TextDecoder();
          let buffer = '';

          const processStream = async () => {
            while (true) {
              const { done, value } = await reader.read();

              if (done) break;

              buffer += decoder.decode(value, { stream: true });

              // Process complete SSE messages
              const lines = buffer.split('\n\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.slice(6)) as ChunkProgressEvent;
                    onProgress(data);

                    if (data.event === 'enhanced_complete' || data.event === 'character_complete' || data.event === 'semantic_complete') {
                      resolve({
                        entries: data.entries || [],
                        global_notes: data.global_notes,
                        session_id: data.session_id,
                        chunks: data.chunks || [],
                        chunk_results: data.chunk_results || [],
                        total_pages: data.total_pages || 0,
                        total_chunks: data.total_chunks || 0,
                        processing_time: data.processing_time || 0,
                        chunk_size: data.chunk_size || 1200,
                      });
                      return;
                    }

                    if (data.event === 'error') {
                      reject(new Error(data.message || 'Processing failed'));
                      return;
                    }
                  } catch (e) {
                    console.error('Failed to parse SSE data:', e);
                  }
                }
              }
            }
          };

          await processStream();
        })
        .catch(reject);
    });
  }



  /**
   * Download JSON file
   */
  downloadJSON(data: any, filename: string = 'extracted_data.json'): void {
    try {
      const jsonString = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      throw new Error('Failed to download JSON file');
    }
  }
}

export const processingService = new ProcessingService();