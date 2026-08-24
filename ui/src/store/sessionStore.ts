import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export const useSessionStore = defineStore('session', () => {
  const messages = ref<Message[]>([]);
  const sessionId = ref<string | null>(null);
  const isGenerating = ref(false);

  const clearMessages = (): void => {
    messages.value = [];
  };

  const addMessage = (role: Message['role'], content: string): void => {
    messages.value.push({
      role,
      content,
      timestamp: new Date(),
    });
  };

  return {
    messages,
    sessionId,
    isGenerating,
    clearMessages,
    addMessage,
  };
});
