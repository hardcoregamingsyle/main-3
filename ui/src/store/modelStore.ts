import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { apiClient } from '../utils/api';

interface ModelInfo {
  id: string;
  name: string;
  size: number;
  status: 'loading' | 'ready' | 'error';
}

export const useModelStore = defineStore('model', () => {
  const models = ref<ModelInfo[]>([]);
  const currentModel = ref<string | null>(null);
  const loading = ref(false);

  const loadModels = async (): Promise<void> => {
    loading.value = true;
    try {
      const data = await apiClient.get<ModelInfo[]>('/models');
      models.value = data;
    } catch (error) {
      console.error('Failed to load models:', error);
    } finally {
      loading.value = false;
    }
  };

  const selectModel = async (modelId: string): Promise<void> => {
    currentModel.value = modelId;
  };

  return {
    models,
    currentModel,
    loading,
    loadModels,
    selectModel,
  };
});
