import { create } from 'zustand';
import type { ProcessingSession, ExtractedData } from '@/types';
import { processingService } from '@/services';

interface ProcessingState {
  sessions: ProcessingSession[];
  currentSession: ProcessingSession | null;
  extractedData: ExtractedData | null;
  isLoading: boolean;
  error: string | null;
}

interface ProcessingActions {
  startProcessing: (fileId: string) => Promise<ProcessingSession>;
  getProcessingStatus: (sessionId: string) => Promise<void>;
  cancelProcessing: (sessionId: string) => Promise<void>;
  getProcessingResult: (sessionId: string) => Promise<void>;
  getSessions: () => Promise<void>;
  setCurrentSession: (session: ProcessingSession | null) => void;
  updateSessionProgress: (sessionId: string, progress: any) => void;
  clearError: () => void;
}

type ProcessingStore = ProcessingState & ProcessingActions;

export const useProcessingStore = create<ProcessingStore>((set, _get) => ({
  // State
  sessions: [],
  currentSession: null,
  extractedData: null,
  isLoading: false,
  error: null,

  // Actions
  startProcessing: async (fileId: string) => {
    set({ isLoading: true, error: null });
    try {
      const session = await processingService.startProcessing(fileId);
      set(state => ({
        sessions: [...state.sessions, session],
        currentSession: session,
        isLoading: false
      }));
      return session;
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to start processing', 
        isLoading: false 
      });
      throw error;
    }
  },

  getProcessingStatus: async (sessionId: string) => {
    try {
      const session = await processingService.getProcessingStatus(sessionId);
      set(state => ({
        sessions: state.sessions.map(s => s.id === sessionId ? session : s),
        currentSession: state.currentSession?.id === sessionId ? session : state.currentSession
      }));
    } catch (error: any) {
      set({ error: error.message || 'Failed to get processing status' });
    }
  },

  cancelProcessing: async (sessionId: string) => {
    set({ isLoading: true, error: null });
    try {
      await processingService.cancelProcessing(sessionId);
      set(state => ({
        sessions: state.sessions.map(s => 
          s.id === sessionId ? { ...s, status: 'error' as const } : s
        ),
        isLoading: false
      }));
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to cancel processing', 
        isLoading: false 
      });
    }
  },

  getProcessingResult: async (sessionId: string) => {
    set({ isLoading: true, error: null });
    try {
      const data = await processingService.getProcessingResult(sessionId);
      set({ extractedData: data, isLoading: false });
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to get processing result', 
        isLoading: false 
      });
    }
  },

  getSessions: async () => {
    set({ isLoading: true, error: null });
    try {
      const sessions = await processingService.getProcessingSessions();
      set({ sessions, isLoading: false });
    } catch (error: any) {
      set({ 
        error: error.message || 'Failed to fetch sessions', 
        isLoading: false 
      });
    }
  },

  setCurrentSession: (session: ProcessingSession | null) => {
    set({ currentSession: session });
  },

  updateSessionProgress: (sessionId: string, progress: any) => {
    set(state => ({
      sessions: state.sessions.map(s => 
        s.id === sessionId ? { ...s, progress } : s
      ),
      currentSession: state.currentSession?.id === sessionId 
        ? { ...state.currentSession, progress } 
        : state.currentSession
    }));
  },

  clearError: () => set({ error: null }),
}));