-- MoE Ultra Engine Database Schema
-- Version: 1.0.0
-- Created: 2026-08-24

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Models table: tracks loaded models and their configurations
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    revision TEXT DEFAULT 'main',
    quantization TEXT,
    max_gpu_memory_gb REAL DEFAULT 0,
    max_cpu_memory_gb REAL DEFAULT 28,
    max_disk_memory_gb REAL DEFAULT 50,
    total_params BIGINT,
    active_params BIGINT,
    num_experts INTEGER,
    experts_per_token INTEGER,
    num_layers INTEGER,
    max_context_length INTEGER,
    architecture TEXT,
    load_time_seconds REAL,
    status TEXT DEFAULT 'unloaded',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_models_model_id ON models(model_id);

-- Generation requests table: tracks all inference requests
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    prompt TEXT NOT NULL,
    max_tokens INTEGER DEFAULT 512,
    temperature REAL DEFAULT 0.7,
    top_p REAL DEFAULT 0.9,
    top_k INTEGER DEFAULT 50,
    repetition_penalty REAL DEFAULT 1.1,
    stream BOOLEAN DEFAULT 1,
    output_text TEXT,
    token_count INTEGER DEFAULT 0,
    generation_time_seconds REAL,
    tokens_per_second REAL,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_generations_model_id ON generations(model_id);
CREATE INDEX IF NOT EXISTS idx_generations_status ON generations(status);
CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at);

-- Benchmarks table: stores performance benchmarks
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    time_to_first_token_ms REAL,
    tokens_per_second REAL,
    peak_memory_gb REAL,
    avg_memory_gb REAL,
    cpu_percent REAL,
    gpu_percent REAL,
    config_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_model_id ON benchmarks(model_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_created_at ON benchmarks(created_at);

-- System metrics table: time-series system resource usage
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_percent REAL,
    ram_used_gb REAL,
    ram_total_gb REAL,
    gpu_percent REAL,
    gpu_memory_used_gb REAL,
    gpu_memory_total_gb REAL,
    disk_used_gb REAL,
    disk_total_gb REAL,
    active_experts INTEGER,
    offloaded_layers INTEGER
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Model layers table: tracks layer offloading status for MoE models
CREATE TABLE IF NOT EXISTS model_layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    layer_index INTEGER NOT NULL,
    layer_type TEXT NOT NULL, -- 'attention', 'mlp', 'moe', 'norm', 'embedding'
    expert_count INTEGER DEFAULT 0,
    active_experts INTEGER DEFAULT 0,
    memory_gb REAL,
    device TEXT, -- 'cuda', 'cpu', 'disk'
    offloaded BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, layer_index)
);

CREATE INDEX IF NOT EXISTS idx_model_layers_model_id ON model_layers(model_id);
CREATE INDEX IF NOT EXISTS idx_model_layers_device ON model_layers(device);

-- Expert activation logs: tracks which experts are activated per token
CREATE TABLE IF NOT EXISTS expert_activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    generation_id INTEGER REFERENCES generations(id) ON DELETE CASCADE,
    layer_index INTEGER NOT NULL,
    token_position INTEGER NOT NULL,
    expert_indices TEXT NOT NULL, -- JSON array of expert indices
    router_logits TEXT, -- JSON array of router logits
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_expert_activations_model_id ON expert_activations(model_id);
CREATE INDEX IF NOT EXISTS idx_expert_activations_generation_id ON expert_activations(generation_id);
CREATE INDEX IF NOT EXISTS idx_expert_activations_layer ON expert_activations(layer_index);

-- Configuration audit log
CREATE TABLE IF NOT EXISTS config_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT DEFAULT 'system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_config_audit_key ON config_audit(key);
CREATE INDEX IF NOT EXISTS idx_config_audit_created_at ON config_audit(created_at);

-- Trigger to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_models_timestamp
AFTER UPDATE ON models
BEGIN
    UPDATE models SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- View for model performance summary
CREATE VIEW IF NOT EXISTS model_performance AS
SELECT
    m.id,
    m.model_id,
    m.quantization,
    m.total_params,
    m.active_params,
    COUNT(g.id) as total_generations,
    AVG(g.tokens_per_second) as avg_tokens_per_second,
    AVG(g.generation_time_seconds) as avg_generation_time,
    MAX(g.tokens_per_second) as max_tokens_per_second,
    MIN(g.generation_time_seconds) as min_generation_time
FROM models m
LEFT JOIN generations g ON m.id = g.model_id AND g.status = 'completed'
GROUP BY m.id;

-- View for recent system health
CREATE VIEW IF NOT EXISTS recent_system_health AS
SELECT
    datetime(timestamp) as time,
    cpu_percent,
    ram_used_gb,
    ram_total_gb,
    gpu_percent,
    gpu_memory_used_gb,
    gpu_memory_total_gb,
    active_experts,
    offloaded_layers
FROM system_metrics
WHERE timestamp > datetime('now', '-1 hour')
ORDER BY timestamp DESC
LIMIT 100;