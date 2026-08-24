// API utilities
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiClient = {
  get: <T>(endpoint: string): Promise<T> => api.get(endpoint).then((res) => res.data),
  post: <T>(endpoint: string, data: unknown): Promise<T> => api.post(endpoint, data).then((res) => res.data),
  put: <T>(endpoint: string, data: unknown): Promise<T> => api.put(endpoint, data).then((res) => res.data),
  delete: <T>(endpoint: string): Promise<T> => api.delete(endpoint).then((res) => res.data),
};
