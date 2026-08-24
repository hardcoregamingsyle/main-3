// MoE Ultra Engine API Client
class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    if (config.body && typeof config.body === 'object') {
      config.body = JSON.stringify(config.body);
    }

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
      }

      return data;
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  // Model management
  async loadModel(config) {
    return this.request('/api/v1/models/load', {
      method: 'POST',
      body: config
    });
  }

  async unloadModel() {
    return this.request('/api/v1/models/unload', {
      method: 'POST'
    });
  }

  async getModelStatus() {
    return this.request('/api/v1/models/status');
  }

  async listModels() {
    return this.request('/api/v1/models/list');
  }

  // Inference
  async generate(request) {
    return this.request('/api/v1/generate', {
      method: 'POST',
      body: request
    });
  }

  async generateStream(request, onToken) {
    const response = await fetch(`${this.baseUrl}/api/v1/generate/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token) onToken(data.token);
            if (data.done) return data;
          } catch (e) {
            console.warn('Failed to parse SSE:', line);
          }
        }
      }
    }
  }

  // System info
  async getSystemInfo() {
    return this.request('/api/v1/system/info');
  }

  async getMemoryStats() {
    return this.request('/api/v1/system/memory');
  }

  async getBenchmarks() {
    return this.request('/api/v1/benchmarks');
  }

  // WebSocket for real-time updates
  connectWebSocket(onMessage, onClose) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.warn('Failed to parse WS message:', event.data);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (onClose) onClose();
      this.attemptReconnect(onMessage, onClose);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  attemptReconnect(onMessage, onClose) {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connectWebSocket(onMessage, onClose);
    }, delay);
  }

  disconnectWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendWebSocketMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ApiClient;
} else {
  window.ApiClient = ApiClient;
}