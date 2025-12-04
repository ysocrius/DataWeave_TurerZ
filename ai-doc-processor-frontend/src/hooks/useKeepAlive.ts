import { useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

/**
 * Hook to keep the backend server alive by pinging it periodically
 * This prevents Render free tier from spinning down after 15 minutes of inactivity
 */
export const useKeepAlive = (intervalMinutes: number = 10) => {
  useEffect(() => {
    // Ping immediately on mount
    const pingServer = async () => {
      try {
        const response = await fetch(`${API_BASE_URL.replace('/api', '')}/api/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (response.ok) {
          console.log('✅ Backend keep-alive ping successful');
        }
      } catch (error) {
        console.warn('⚠️ Backend keep-alive ping failed:', error);
      }
    };

    // Initial ping
    pingServer();

    // Set up interval to ping every X minutes
    const intervalId = setInterval(pingServer, intervalMinutes * 60 * 1000);

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, [intervalMinutes]);
};
