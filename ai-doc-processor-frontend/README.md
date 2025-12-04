# AI Document Processor - Frontend

A modern React frontend for the AI-Powered Document Processor, built with TypeScript, Vite, and Mantine UI.

## Features

- **Modern React 18** with TypeScript for type safety
- **Vite** for fast development and building
- **Mantine UI** for consistent, accessible components
- **React Query** for server state management
- **Zustand** for client state management
- **React Router** for navigation
- **Socket.IO** for real-time updates
- **ESLint & Prettier** for code quality

## Project Structure

```
src/
├── components/     # Reusable UI components
├── hooks/         # Custom React hooks
├── pages/         # Page components
├── services/      # API services and HTTP client
├── store/         # Zustand stores for state management
├── types/         # TypeScript type definitions
├── utils/         # Utility functions
├── App.tsx        # Main application component
├── main.tsx       # Application entry point
└── index.css      # Global styles
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Update environment variables in `.env` as needed

### Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Building

Build for production:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting
- `npm run type-check` - Run TypeScript type checking

## Architecture

### State Management

- **Zustand** for client-side state (UI state, local data)
- **React Query** for server state (API data, caching, synchronization)

### API Communication

- Custom API client with automatic token management
- Automatic retry logic with exponential backoff
- File upload with progress tracking
- WebSocket integration for real-time updates

### Routing

- React Router for client-side routing
- Protected routes with authentication guards
- Lazy loading for code splitting

### Styling

- Mantine UI component library
- CSS custom properties for theming
- Responsive design utilities
- Dark/light mode support

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000/api` |
| `VITE_WS_URL` | WebSocket server URL | `http://localhost:8000` |
| `VITE_APP_NAME` | Application name | `AI Document Processor` |
| `VITE_ENABLE_DEBUG` | Enable debug mode | `true` |

## Development Guidelines

### Code Style

- Use TypeScript for all new code
- Follow ESLint and Prettier configurations
- Use functional components with hooks
- Prefer composition over inheritance

### State Management

- Use Zustand for local/UI state
- Use React Query for server state
- Keep stores focused and small
- Use TypeScript for store typing

### Component Development

- Create reusable components in `/components`
- Use Mantine UI components as base
- Implement proper TypeScript props
- Add proper accessibility attributes

### API Integration

- Use the provided API client
- Handle loading and error states
- Implement proper error boundaries
- Use React Query for caching

## Next Steps

This is the initial project structure for Task 1.1. The following tasks will implement:

- Authentication system (Task 2.3)
- File upload components (Task 4.3)
- Real-time WebSocket integration (Task 5.5)
- Data visualization components (Task 9.1-9.5)
- Progressive Web App features (Task 11.1-11.5)

## Contributing

1. Follow the established code style
2. Write TypeScript for all new code
3. Add proper error handling
4. Test components thoroughly
5. Update documentation as needed