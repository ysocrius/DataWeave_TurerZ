import { useState, useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

export type BackendStatus = 'checking' | 'online' | 'offline' | 'error';

interface BackendHealth {
  status: string;
  api: string;
  timestamp: number;
  learning_enabled: boolean;
  learning_connected: boolean;
}

/**
 * Hook to monitor backend server status
 * Returns real-time status of the backend API
 */
export const useBackendStatus = (checkInterval: number = 30000) => {
  const [status, setStatus] = useState<BackendStatus>('checking');
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL.replace('/api', '')}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add timeout to prevent hanging
        signal: AbortSignal.timeout(10000), // 10 second timeout
      });

      if (response.ok) {
        const healthData: BackendHealth = await response.json();
        setHealth(healthData);
        setStatus('online');
        setLastCheck(new Date());
      } else {
        setStatus('error');
        setHealth(null);
      }
    } catch (error) {
      console.warn('Backend health check failed:', error);
      setStatus('offline');
      setHealth(null);
    }
  };

  useEffect(() => {
    // Initial check
    checkBackendHealth();

    // Set up interval for periodic checks
    const intervalId = setInterval(checkBackendHealth, checkInterval);

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, [checkInterval]);

  // Manual refresh function
  const refresh = () => {
    setStatus('checking');
    checkBackendHealth();
  };

  return {
    status,
    health,
    lastCheck,
    refresh,
    isOnline: status === 'online',
    isOffline: status === 'offline' || status === 'error',
  };
};