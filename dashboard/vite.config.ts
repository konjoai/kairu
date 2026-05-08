import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    dedupe: ['react', 'react-dom', 'motion', 'motion/react'],
  },
  server: {
    port: 5176,
    proxy: {
      // Real /generate (and /metrics, /health) live on the FastAPI server.
      '/generate':  { target: 'http://localhost:8000', changeOrigin: true },
      '/metrics':   { target: 'http://localhost:8000', changeOrigin: true },
      '/health':    { target: 'http://localhost:8000', changeOrigin: true },
      // Speculative simulation lives on the demo stdlib server.
      '/api':       { target: 'http://localhost:7777', changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
