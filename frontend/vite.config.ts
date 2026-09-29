/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The FastAPI backend (backend/api). Same origin for the browser, so no CORS.
    // start.sh sets API_PORT when the API runs somewhere other than 8000.
    proxy: { '/api': `http://localhost:${process.env.API_PORT ?? '8000'}` },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
