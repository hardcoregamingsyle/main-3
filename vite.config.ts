import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './ui/src'),
      '@components': path.resolve(__dirname, './ui/src/components'),
      '@utils': path.resolve(__dirname, './ui/src/utils'),
      '@store': path.resolve(__dirname, './ui/src/store')
    }
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
          axios: ['axios']
        }
      }
    }
  },
  css: {
    preprocessorOptions: {
      scss: {
        includePaths: ['./ui/src/styles']
      }
    }
  }
})