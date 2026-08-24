-- MoE Ultra Engine Database Schema
-- Version: 1.0.0
-- Date: 2026-08-24

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    path VARCHAR(512) NOT NULL UNIQUE,
    size BIGINT NOT NULL,
    format VARCHAR(50) NOT NULL,
    quantization_level INTEGER DEFAULT 0,
    parameters_total NUMERIC(20,2),
    parameters_active NUMERIC(20,2),
    expert_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'available',
    checksum VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    name VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2048,
    top_p REAL DEFAULT 0.95,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    total_tokens INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    order_index INTEGER
);

-- Create inference metrics table
CREATE TABLE IF NOT EXISTS inference_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tokens_per_second REAL,
    time_to_first_token_ms INTEGER,
    total_inference_time_ms INTEGER,
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    gpu_usage_percent REAL,
    batch_size INTEGER,
    context_window INTEGER,
    expert_activation_ratio REAL
);

-- Create system settings table
CREATE TABLE IF NOT EXISTS system_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON inference_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON inference_metrics(timestamp);

-- Insert default system settings
INSERT INTO system_settings (id, setting_key, setting_value)
VALUES 
    (1, '{"max_concurrent_sessions": 10}', '{"default_temperature": 0.7, "max_context_length": 32768, "batch_size": 4}');

-- Add comments for documentation
COMMENT ON TABLE models IS 'Model registry and metadata storage';
COMMENT ON TABLE sessions IS 'Chat conversation sessions';
COMMENT ON TABLE messages IS 'Individual chat messages within sessions';
COMMENT ON TABLE inference_metrics IS 'Performance metrics for inference runs';
COMMENT ON TABLE system_settings IS 'Global system configuration';
