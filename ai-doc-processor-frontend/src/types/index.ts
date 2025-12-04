// TypeScript type definitions

// User types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'user' | 'admin';
  preferences: UserPreferences;
  createdAt: Date;
  lastLoginAt: Date;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  notifications: boolean;
  autoSave: boolean;
}

// File types
export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'uploading' | 'uploaded' | 'processing' | 'completed' | 'error';
  progress: number;
  uploadedAt: Date;
  processedAt?: Date;
  error?: string;
}

// Processing types
export interface ProcessingSession {
  id: string;
  fileId: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress: ProcessingProgress;
  result?: ExtractedData;
  error?: ProcessingError;
  createdAt: Date;
  completedAt?: Date;
}

export interface ProcessingProgress {
  step: string;
  percentage: number;
  message: string;
  estimatedTimeRemaining?: number;
}

export interface ProcessingError {
  code: string;
  message: string;
  details?: string;
}

// Data types
export interface ExtractedData {
  entries: DataEntry[];
  globalNotes?: string;
  metadata: ExtractionMetadata;
}

export interface DataEntry {
  id: string;
  [key: string]: any; // Dynamic fields based on extraction
}

export interface ExtractionMetadata {
  totalEntries: number;
  columnsDetected: string[];
  confidence: number;
  processingTime: number;
}

// API types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  success: boolean;
}

export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

// Authentication types
export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

// WebSocket types
export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: Date;
}

export interface ProgressUpdate {
  sessionId: string;
  progress: ProcessingProgress;
}

// Export types
export interface ExportOptions {
  format: 'excel' | 'csv' | 'json' | 'pdf';
  fields?: string[];
  template?: string;
  customization?: Record<string, any>;
}

export interface ExportJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  options: ExportOptions;
  downloadUrl?: string;
  createdAt: Date;
  completedAt?: Date;
}