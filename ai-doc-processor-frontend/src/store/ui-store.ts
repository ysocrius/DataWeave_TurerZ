import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  theme: 'light' | 'dark' | 'auto';
  sidebarOpen: boolean;
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message: string;
    timestamp: Date;
    autoClose?: boolean;
  }>;
  isLoading: boolean;
  loadingMessage: string;
}

interface UIActions {
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  addNotification: (notification: Omit<UIState['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  setLoading: (loading: boolean, message?: string) => void;
}

type UIStore = UIState & UIActions;

export const useUIStore = create<UIStore>()(
  persist(
    (set, get) => ({
      // State
      theme: 'auto',
      sidebarOpen: true,
      notifications: [],
      isLoading: false,
      loadingMessage: '',

      // Actions
      setTheme: (theme: 'light' | 'dark' | 'auto') => {
        set({ theme });
      },

      toggleSidebar: () => {
        set(state => ({ sidebarOpen: !state.sidebarOpen }));
      },

      setSidebarOpen: (open: boolean) => {
        set({ sidebarOpen: open });
      },

      addNotification: (notification) => {
        const id = Math.random().toString(36).substr(2, 9);
        const newNotification = {
          ...notification,
          id,
          timestamp: new Date(),
        };

        set(state => ({
          notifications: [...state.notifications, newNotification]
        }));

        // Auto-remove notification after 5 seconds if autoClose is true
        if (notification.autoClose !== false) {
          setTimeout(() => {
            get().removeNotification(id);
          }, 5000);
        }
      },

      removeNotification: (id: string) => {
        set(state => ({
          notifications: state.notifications.filter(n => n.id !== id)
        }));
      },

      clearNotifications: () => {
        set({ notifications: [] });
      },

      setLoading: (loading: boolean, message: string = '') => {
        set({ isLoading: loading, loadingMessage: message });
      },
    }),
    {
      name: 'ui-store',
      partialize: (state) => ({ 
        theme: state.theme,
        sidebarOpen: state.sidebarOpen 
      }),
    }
  )
);