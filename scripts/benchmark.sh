#!/bin/bash
# MoE Ultra Engine Benchmark Script
# Runs comprehensive benchmarks on loaded models

set -euo pipefail

# Configuration
MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B}"
QUANTIZATION="${QUANTIZATION:-int4}"
MAX_GPU_GB="${MAX_GPU_GB:-4}"
MAX_CPU_GB="${MAX_CPU_GB:-28}"
PROMPTS_FILE="${PROMPTS_FILE:-benchmarks/prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/results}"
NUM_RUNS="${NUM_RUNS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.json"

log_info "Starting benchmark for $MODEL_ID ($QUANTIZATION)"
log_info "Results will be saved to $RESULTS_FILE"

# Check if server is running
if ! curl -s http://localhost:3000/health > /dev/null; then
    log_error "Server not running on localhost:3000"
    log_info "Start server with: python -m api.main"
    exit 1
fi

# Load model if not already loaded
log_info "Loading model..."
LOAD_RESPONSE=$(curl -s -X POST http://localhost:3000/v1/models/load \
    -H "Content-Type: application/json" \
    -d "{\"model_id\": \"$MODEL_ID\", \"quantization\": \"$QUANTIZATION\", \"max_gpu_memory_gb\": $MAX_GPU_GB, \"max_cpu_memory_gb\": $MAX_CPU_GB}")

echo "$LOAD_RESPONSE" | jq .

# Wait for model to load
sleep 10

# Default prompts if file doesn't exist
if [[ ! -f "$PROMPTS_FILE" ]]; then
    log_warn "Prompts file not found, using defaults"
    cat > "$PROMPTS_FILE" << 'EOF'
Explain quantum computing in simple terms.
Write a Python function to calculate fibonacci numbers.
What are the key differences between transformers and RNNs?
Describe the architecture of a Mixture of Experts model.
Summarize the benefits of model quantization.
EOF
fi

# Run benchmarks
log_info "Running $WARMUP_RUNS warmup runs..."
for i in $(seq 1 $WARMUP_RUNS); do
    prompt=$(head -n 1 "$PROMPTS_FILE")
    curl -s -X POST http://localhost:3000/v1/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_new_tokens\": 100, \"stream\": false}" > /dev/null
    echo -n "."
done
echo ""

log_info "Running $NUM_RUNS benchmark runs..."

# Initialize results JSON
echo '{"model": "'$MODEL_ID'", "quantization": "'$QUANTIZATION'", "runs": []}' > "$RESULTS_FILE"

run_benchmark() {
    local prompt="$1"
    local run_num="$2"
    
    log_info "Run $run_num: ${prompt:0:50}..."
    
    START_TIME=$(date +%s.%3N)
    RESPONSE=$(curl -s -X POST http://localhost:3000/v1/generate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_new_tokens\": 256, \"stream\": false, \"temperature\": 0.7}")
    END_TIME=$(date +%s.%3N)
    
    # Parse response
    GENERATED_TEXT=$(echo "$RESPONSE" | jq -r '.generated_text // empty')
    TOKENS_GENERATED=$(echo "$RESPONSE" | jq -r '.tokens_generated // 0')
    GEN_TIME_MS=$(echo "$RESPONSE" | jq -r '.generation_time_ms // 0')
    TPS=$(echo "$RESPONSE" | jq -r '.tokens_per_second // 0')
    FINISH_REASON=$(echo "$RESPONSE" | jq -r '.finish_reason // "unknown"')
    
    # Calculate wall time
    WALL_TIME_MS=$(echo "($END_TIME - $START_TIME) * 1000" | bc -l)
    
    # Append to results
    jq --argjson run "{\"run\": $run_num, \"prompt\": \"$prompt\", \"generated_text\": \"$GENERATED_TEXT\", \"tokens_generated\": $TOKENS_GENERATED, \"generation_time_ms\": $GEN_TIME_MS, \"wall_time_ms\": $WALL_TIME_MS, \"tokens_per_second\": $TPS, \"finish_reason\": \"$FINISH_REASON\"}" '
    .runs += [$run]' "$RESULTS_FILE" > "$RESULTS_FILE.tmp" && mv "$RESULTS_FILE.tmp" "$RESULTS_FILE"
}

# Read prompts and run benchmarks
run_num=0
while IFS= read -r prompt; do
    [[ -z "$prompt" ]] && continue
    [[ "$prompt" =~ ^#.* ]] && continue
    
    run_num=$((run_num + 1))
    if [[ $run_num -gt $NUM_RUNS ]]; then
        break
    fi
    
    run_benchmark "$prompt" "$run_num"
    sleep 2

done < "$PROMPTS_FILE"

# Calculate summary statistics
log_info "Calculating summary..."
TOTAL_TOKENS=$(jq '[.runs[].tokens_generated] | add' "$RESULTS_FILE")
TOTAL_TIME=$(jq '[.runs[].generation_time_ms] | add' "$RESULTS_FILE")
AVG_TPS=$(echo "scale=2; $TOTAL_TOKENS / ($TOTAL_TIME / 1000)" | bc -l)
MIN_TPS=$(jq '[.runs[].tokens_per_second] | min' "$RESULTS_FILE")
MAX_TPS=$(jq '[.runs[].tokens_per_second] | max' "$RESULTS_FILE")

# Add summary to results
jq --argjson total_tokens "$TOTAL_TOKENS" \
   --argjson total_time "$TOTAL_TIME" \
   --argjson avg_tps "$AVG_TPS" \
   --argjson min_tps "$MIN_TPS" \
   --argjson max_tps "$MAX_TPS" \
   '.summary = {
       "total_tokens": $total_tokens,
       "total_time_ms": $total_time,
       "avg_tokens_per_second": $avg_tps,
       "min_tokens_per_second": $min_tps,
       "max_tokens_per_second": $max_tps,
       "num_runs": (.runs | length)
   }' "$RESULTS_FILE" > "$RESULTS_FILE.tmp" && mv "$RESULTS_FILE.tmp" "$RESULTS_FILE"

log_info "Benchmark complete!"
log_info "Average throughput: ${AVG_TPS} tokens/sec"
log_info "Min throughput: ${MIN_TPS} tokens/sec"
log_info "Max throughput: ${MAX_TPS} tokens/sec"
log_info "Results saved to $RESULTS_FILE"

# Print summary
cat "$RESULTS_FILE" | jq '.summary'
