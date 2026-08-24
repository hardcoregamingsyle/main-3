import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './ui/src'),
      '@components': path.resolve(__dirname, './ui/src/components'),
      '@utils': path.resolve(__dirname, './ui/src/utils'),
      '@store': path.resolve(__dirname, './ui/src/store'),
      '@assets': path.resolve(__dirname, './ui/static'),
    },
  },
  root: '.',
  base: '/',
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'esbuild',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'ui/index.html'),
      },
    },
  },
});