<template>
  <header class="header">
    <div class="header-container">
      <h1 class="logo">MoE Ultra Engine</h1>
      <nav v-if="showNav" class="nav">
        <a href="#dashboard" class="nav-link">Dashboard</a>
        <a href="#models" class="nav-link">Models</a>
        <a href="#settings" class="nav-link">Settings</a>
      </nav>
      <div class="status-indicator">
        <span class="status-dot" :class="statusClass"></span>
        <span class="status-text">{{ statusText }}</span>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const showNav = ref(true)
const statusClass = ref('online')
const statusText = ref('Ready')

const updateStatus = () => {
  // Mock status update
  statusClass.value = 'online'
  statusText.value = 'System Ready'
}

// Initialize status
updateStatus()
</script>

<style scoped>
.header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  padding: 1rem 2rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.header-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
}

.logo {
  font-size: 1.5rem;
  font-weight: bold;
  color: #00d9ff;
  margin: 0;
}

.nav {
  display: flex;
  gap: 2rem;
}

.nav-link {
  color: #ffffff80;
  text-decoration: none;
  transition: color 0.3s ease;
  font-weight: 500;
}

.nav-link:hover {
  color: #00d9ff;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #00ff88;
}

.status-dot.online {
  background-color: #00ff88;
  box-shadow: 0 0 10px #00ff88;
}

.status-dot.offline {
  background-color: #ff4444;
  box-shadow: 0 0 10px #ff4444;
}

.status-dot.warning {
  background-color: #ffaa00;
  box-shadow: 0 0 10px #ffaa00;
}

.status-text {
  color: #ffffff80;
  font-size: 0.875rem;
}
</style>
