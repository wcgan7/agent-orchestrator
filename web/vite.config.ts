import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8080'
const wsProxyTarget = process.env.VITE_WS_PROXY_TARGET || apiProxyTarget.replace(/^http/, 'ws')
const devPort = Number(process.env.VITE_PORT || 3000)

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/react') || id.includes('node_modules/scheduler')) return 'framework'
          if (id.includes('node_modules/@xyflow')) return 'graph'
          if (id.includes('/src/components/KanbanBoard/')) return 'kanban'
          if (id.includes('/src/components/AgentCard/')) return 'agents'
        },
      },
    },
  },
  server: {
    port: devPort,
    host: true,
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      '/ws': {
        target: wsProxyTarget,
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
