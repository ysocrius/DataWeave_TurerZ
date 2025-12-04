import { Routes, Route } from 'react-router-dom';
import { MantineProvider } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { DashboardPage } from '@/pages/DashboardPage';
import { theme } from './theme';
import '@mantine/core/styles.css';
import '@mantine/dropzone/styles.css';
import '@mantine/notifications/styles.css';

function App() {
  return (
    <MantineProvider theme={theme}>
      <Notifications position="top-right" />
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
      </Routes>
    </MantineProvider>
  );
}

export default App;