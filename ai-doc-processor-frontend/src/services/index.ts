// Export all services
export { default as authService } from './auth-service';
export { default as fileService } from './file-service';
export { processingService } from './processing-service';
export type { 
    ProcessingResult, 
    ProcessingJob, 
    ChunkedProcessingResult, 
    ChunkInfo,
    ChunkProgressEvent,
    ChunkProgressCallback 
} from './processing-service';