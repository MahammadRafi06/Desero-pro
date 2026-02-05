# Parallelism Strategy Advisor - FastAPI Server

A comprehensive REST API for distributed training parallelism optimization, providing programmatic access to all parallelism planning features.

## Features

- **Parallelism Recommendations**: Get optimal TP/PP/DP/CP/EP strategies
- **Sharding Strategies**: ZeRO-1/2/3, FSDP, and HSDP recommendations
- **Launch Commands**: Generate torchrun, DeepSpeed, Accelerate, and Megatron commands
- **Cost Analysis**: Pareto frontier analysis for cost vs throughput tradeoffs
- **Model Analysis**: Detailed model architecture and memory analysis
- **Inference Optimization**: Quantization, KV-cache, and speculative decoding
- **RLHF Support**: Memory calculations and optimization for PPO/DPO/RLOO
- **MoE Optimization**: Mixture of Experts parallelism strategies
- **Long Context**: Optimization for extended sequence lengths
- **vLLM Integration**: Configuration generation for vLLM serving
- **Troubleshooting**: Error diagnosis and NCCL tuning
- **Large Scale**: Multi-node cluster optimization (100s to 1000s of GPUs)

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Additional dependencies for the API server
pip install fastapi uvicorn pydantic
```

### Start the Server

```bash
# Method 1: Using Python directly
python parallelism_planner_server.py

# Method 2: Using uvicorn
uvicorn parallelism_planner_server:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Production with multiple workers
uvicorn parallelism_planner_server:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Core Features

#### 1. Get Parallelism Recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 1,
    "seq_length": 2048,
    "goal": "throughput",
    "training": true,
    "mock_topology": "h100",
    "mock_gpus": 8
  }'
```

#### 2. Get Sharding Strategy

```bash
curl -X POST "http://localhost:8000/sharding" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "dp": 8,
    "memory": 80,
    "batch_size": 1,
    "seq_length": 2048
  }'
```

#### 3. Generate Launch Commands

```bash
curl -X POST "http://localhost:8000/launch" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": 2,
    "gpus": 8,
    "tp": 2,
    "pp": 2,
    "dp": 4,
    "sharding": "zero3",
    "micro_batch": 1,
    "grad_accum": 16,
    "script": "train.py"
  }'
```

#### 4. Pareto Analysis (Cost vs Throughput)

```bash
curl -X POST "http://localhost:8000/pareto" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "gpu_cost": 4.0,
    "batch_size": 8,
    "seq_length": 4096,
    "training": true
  }'
```

#### 5. Analyze Model Architecture

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 8,
    "seq_length": 2048
  }'
```

#### 6. Estimate Training Time and Cost

```bash
curl -X POST "http://localhost:8000/estimate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "tokens": 1000000000000,
    "throughput": 100000,
    "gpus": 64,
    "gpu_cost": 4.0,
    "checkpoint_interval": 1000000000
  }'
```

### Advanced Features

#### 7. Compare Multiple Models

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B"],
    "batch_size": 1,
    "seq_length": 2048,
    "training": true,
    "mock_topology": "h100"
  }'
```

#### 8. Validate Configuration

```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "tp": 2,
    "pp": 2,
    "dp": 4,
    "memory": 80,
    "batch_size": 1,
    "seq_length": 2048
  }'
```

#### 9. Advanced Optimization

```bash
curl -X POST "http://localhost:8000/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 1,
    "seq_length": 2048,
    "training": true,
    "precision": "bf16",
    "checkpointing": "selective"
  }'
```

#### 10. Bottleneck Analysis

```bash
curl -X POST "http://localhost:8000/bottleneck" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "tp": 2,
    "pp": 2,
    "dp": 4,
    "batch_size": 8,
    "seq_length": 2048
  }'
```

#### 11. Scaling Analysis

```bash
curl -X POST "http://localhost:8000/scaling" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "base_gpus": 8,
    "target_gpus": 64,
    "batch_size": 8,
    "seq_length": 2048
  }'
```

#### 12. What-If Analysis

```bash
curl -X POST "http://localhost:8000/whatif" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "scenario": "double_batch_size",
    "current_tp": 2,
    "current_pp": 2,
    "current_dp": 4,
    "current_batch_size": 8
  }'
```

#### 13. Find Maximum Batch Size

```bash
curl -X POST "http://localhost:8000/batchsize" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "memory": 80,
    "seq_length": 2048,
    "tp": 2,
    "pp": 2
  }'
```

#### 14. Auto-Tune Configuration

```bash
curl -X POST "http://localhost:8000/autotune" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "gpus": 8,
    "memory": 80,
    "seq_length": 2048,
    "training": true
  }'
```

### Inference Optimization

#### 15. Inference Optimization

```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 1,
    "seq_length": 2048,
    "max_output_length": 512,
    "latency_target_ms": 100,
    "mock_topology": "h100"
  }'
```

#### 16. vLLM Configuration

```bash
curl -X POST "http://localhost:8000/vllm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 64,
    "max_tokens": 512,
    "throughput_target": 10000,
    "gpus": 2
  }'
```

### Advanced Training

#### 17. RLHF Optimization

```bash
curl -X POST "http://localhost:8000/rlhf" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "algorithm": "ppo",
    "batch_size": 8,
    "seq_length": 2048,
    "gpus": 16,
    "memory": 80
  }'
```

#### 18. MoE (Mixture of Experts) Optimization

```bash
curl -X POST "http://localhost:8000/moe" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mixtral-8x7b",
    "num_experts": 8,
    "experts_per_token": 2,
    "batch_size": 8,
    "seq_length": 2048,
    "gpus": 8
  }'
```

#### 19. Long Context Optimization

```bash
curl -X POST "http://localhost:8000/long-context" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "seq_length": 32768,
    "batch_size": 1,
    "gpus": 8,
    "memory": 80
  }'
```

#### 20. Communication Overlap Analysis

```bash
curl -X POST "http://localhost:8000/comm-overlap" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "tp": 2,
    "pp": 2,
    "dp": 4,
    "batch_size": 8
  }'
```

### Troubleshooting & Utilities

#### 21. Troubleshoot Issues

```bash
# Get all troubleshooting topics
curl "http://localhost:8000/troubleshoot" \
  -H "Content-Type: application/json" \
  -d '{}'

# Diagnose specific error
curl -X POST "http://localhost:8000/troubleshoot" \
  -H "Content-Type: application/json" \
  -d '{
    "error_message": "CUDA out of memory"
  }'

# Get specific topic info
curl -X POST "http://localhost:8000/troubleshoot" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "oom"
  }'
```

#### 22. Memory Breakdown

```bash
curl -X POST "http://localhost:8000/memory" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 8,
    "seq_length": 2048,
    "tp": 2,
    "pp": 2,
    "training": true
  }'
```

#### 23. NCCL Tuning

```bash
curl -X POST "http://localhost:8000/nccl" \
  -H "Content-Type: application/json" \
  -d '{
    "num_nodes": 4,
    "gpus_per_node": 8,
    "network_type": "nvlink"
  }'
```

#### 24. Export Configuration

```bash
curl -X POST "http://localhost:8000/export" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "tp": 2,
    "pp": 2,
    "dp": 4,
    "sharding": "zero3",
    "batch_size": 1,
    "seq_length": 2048,
    "framework": "pytorch"
  }'
```

#### 25. Large-Scale Cluster Optimization

```bash
curl -X POST "http://localhost:8000/largescale" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-405B",
    "num_nodes": 64,
    "gpus_per_node": 8,
    "batch_size": 8,
    "seq_length": 4096,
    "cluster_type": "nvlink_h100"
  }'
```

### Utility Endpoints

#### 26. List Model Presets

```bash
curl "http://localhost:8000/presets"
```

#### 27. Get Hardware Topology

```bash
# Auto-detect topology
curl "http://localhost:8000/topology"

# Use mock topology
curl "http://localhost:8000/topology?mock=h100&mock_gpus=8"
```

#### 28. Health Check

```bash
curl "http://localhost:8000/health"
```

## Python Client Example

```python
import requests
import json

# Base URL
API_URL = "http://localhost:8000"

# Example: Get parallelism recommendations
response = requests.post(
    f"{API_URL}/recommend",
    json={
        "model": "meta-llama/Llama-3.1-70B",
        "batch_size": 1,
        "seq_length": 2048,
        "goal": "throughput",
        "training": True,
        "mock_topology": "h100",
        "mock_gpus": 8
    }
)

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.json())

# Example: Get sharding recommendations
response = requests.post(
    f"{API_URL}/sharding",
    json={
        "model": "meta-llama/Llama-3.1-70B",
        "dp": 8,
        "memory": 80,
        "batch_size": 1,
        "seq_length": 2048
    }
)

result = response.json()
print(json.dumps(result, indent=2))

# Example: Generate launch commands
response = requests.post(
    f"{API_URL}/launch",
    json={
        "nodes": 2,
        "gpus": 8,
        "tp": 2,
        "pp": 2,
        "dp": 4,
        "sharding": "zero3",
        "script": "train.py"
    }
)

result = response.json()
print(result["commands"]["torchrun"]["command"])
```

## JavaScript/TypeScript Client Example

```javascript
const API_URL = "http://localhost:8000";

// Example: Get parallelism recommendations
async function getRecommendations() {
  const response = await fetch(`${API_URL}/recommend`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "meta-llama/Llama-3.1-70B",
      batch_size: 1,
      seq_length: 2048,
      goal: "throughput",
      training: true,
      mock_topology: "h100",
      mock_gpus: 8,
    }),
  });

  if (response.ok) {
    const result = await response.json();
    console.log(result);
  } else {
    console.error("Error:", response.status);
  }
}

// Example: Get launch commands
async function getLaunchCommands() {
  const response = await fetch(`${API_URL}/launch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      nodes: 2,
      gpus: 8,
      tp: 2,
      pp: 2,
      dp: 4,
      sharding: "zero3",
      script: "train.py",
    }),
  });

  const result = await response.json();
  console.log(result.commands.torchrun.command);
}

getRecommendations();
getLaunchCommands();
```

## API Response Format

All successful responses return JSON with appropriate data structures. Error responses follow this format:

```json
{
  "error": "Error message",
  "type": "ExceptionType",
  "traceback": "Full traceback for debugging"
}
```

## Configuration

### Environment Variables

You can configure the server using environment variables:

```bash
# Server configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"

# GPU topology (if not auto-detected)
export MOCK_TOPOLOGY="h100"
export MOCK_GPUS="8"
```

### Production Deployment

For production deployment, consider using:

```bash
# With Gunicorn
gunicorn parallelism_planner_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# With systemd service
# Create /etc/systemd/system/parallelism-api.service
[Unit]
Description=Parallelism Strategy Advisor API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/path/to/Desero-pro
ExecStart=/usr/bin/uvicorn parallelism_planner_server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic

COPY . .

EXPOSE 8000

CMD ["uvicorn", "parallelism_planner_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t parallelism-api .
docker run -p 8000:8000 parallelism-api
```

## API Limits and Best Practices

1. **Rate Limiting**: Consider implementing rate limiting for production use
2. **Timeouts**: Some operations (especially large model analysis) may take time
3. **Caching**: Results can be cached for identical requests
4. **Authentication**: Add authentication for production deployments
5. **CORS**: Configure CORS headers if accessing from web browsers

## Troubleshooting

### Server won't start

```bash
# Check if port is already in use
lsof -i :8000

# Use a different port
uvicorn parallelism_planner_server:app --port 8001
```

### Import errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install fastapi uvicorn pydantic

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### GPU detection fails

The API automatically falls back to mock topologies. You can specify mock topology in requests:

```json
{
  "mock_topology": "h100",
  "mock_gpus": 8
}
```

## Support

For issues, feature requests, or questions:
- Check the interactive API documentation at `/docs`
- Review the error messages and tracebacks in responses
- Ensure all dependencies are correctly installed

## License

See the main project LICENSE file.

