# Parallelism Strategy Advisor API - Quick Reference Card

## 🚀 Start Server

```bash
# Quick start
./start_api.sh

# Production mode
./start_api.sh --prod --workers 4

# Custom port
./start_api.sh --port 8080

# Docker
docker-compose up -d
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📋 Common Endpoints

### Get Recommendations
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "training": true, "mock_topology": "h100"}'
```

### Generate Launch Commands
```bash
curl -X POST http://localhost:8000/launch \
  -H "Content-Type: application/json" \
  -d '{"nodes": 2, "gpus": 8, "tp": 2, "pp": 2, "dp": 4, "sharding": "zero3", "script": "train.py"}'
```

### Validate Configuration
```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "tp": 2, "pp": 2, "dp": 4, "memory": 80}'
```

### Estimate Cost
```bash
curl -X POST http://localhost:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "tokens": 1000000000000, "throughput": 100000, "gpus": 64}'
```

### Get Memory Breakdown
```bash
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "batch_size": 8, "tp": 2, "pp": 2, "training": true}'
```

### Troubleshoot Error
```bash
curl -X POST http://localhost:8000/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"error_message": "CUDA out of memory"}'
```

### Inference Optimization
```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "latency_target_ms": 100}'
```

### vLLM Configuration
```bash
curl -X POST http://localhost:8000/vllm \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "batch_size": 64, "gpus": 2}'
```

## 🐍 Python Client

```python
from parallelism_api_client import ParallelismAPIClient

client = ParallelismAPIClient("http://localhost:8000")

# Get recommendations
result = client.recommend(
    model="meta-llama/Llama-3.1-70B",
    training=True,
    mock_topology="h100"
)

# Generate launch commands
commands = client.launch(
    nodes=2, gpus=8, tp=2, pp=2, dp=4,
    sharding="zero3", script="train.py"
)

# Validate configuration
validation = client.validate(
    model="meta-llama/Llama-3.1-70B",
    tp=2, pp=2, dp=4, memory=80
)
```

## 🔧 Testing

```bash
# Run test suite
python test_parallelism_api.py

# Run examples
python examples_api_usage.py

# Health check
curl http://localhost:8000/health
```

## 📦 All Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/presets` | GET | List model presets |
| `/topology` | GET | Hardware topology |
| `/recommend` | POST | Parallelism recommendations |
| `/sharding` | POST | Sharding strategies |
| `/launch` | POST | Launch commands |
| `/pareto` | POST | Pareto analysis |
| `/analyze` | POST | Model analysis |
| `/estimate` | POST | Cost estimation |
| `/compare` | POST | Compare models |
| `/validate` | POST | Validate config |
| `/optimize` | POST | Advanced optimization |
| `/bottleneck` | POST | Bottleneck analysis |
| `/scaling` | POST | Scaling analysis |
| `/whatif` | POST | What-if scenarios |
| `/batchsize` | POST | Max batch size |
| `/autotune` | POST | Auto-tune config |
| `/inference` | POST | Inference optimization |
| `/troubleshoot` | POST | Troubleshooting |
| `/memory` | POST | Memory breakdown |
| `/rlhf` | POST | RLHF optimization |
| `/moe` | POST | MoE optimization |
| `/long-context` | POST | Long context |
| `/vllm` | POST | vLLM config |
| `/comm-overlap` | POST | Comm overlap |
| `/export` | POST | Export config |
| `/largescale` | POST | Large-scale |
| `/rl` | POST | RL optimization |
| `/nccl` | POST | NCCL tuning |

## 📚 Documentation

- **Quick Start:** `QUICKSTART_API.md`
- **Full API Docs:** `PARALLELISM_API_README.md`
- **Summary:** `FASTAPI_CONVERSION_SUMMARY.md`
- **Interactive:** http://localhost:8000/docs

## 🐳 Docker Commands

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f parallelism-api

# Stop
docker-compose down

# Rebuild
docker-compose up -d --build

# With nginx (production)
docker-compose --profile production up -d
```

## 🔒 Production Deployment

```bash
# SystemD service
sudo systemctl start parallelism-api
sudo systemctl status parallelism-api
sudo systemctl logs -f parallelism-api

# With Gunicorn
gunicorn parallelism_planner_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 🛠️ Troubleshooting

```bash
# Port already in use
lsof -i :8000
# Use different port: --port 8080

# Check dependencies
pip install -r requirements.txt
pip install fastapi uvicorn pydantic

# Connection refused
curl http://localhost:8000/health
# Ensure server is running

# View server logs
docker-compose logs -f
# or
journalctl -u parallelism-api -f
```

## 📞 Common Workflows

### Workflow 1: Training Setup
```bash
# 1. Get recommendations
curl -X POST localhost:8000/recommend -d '{"model":"llama-3.1-70b","training":true}'

# 2. Validate config
curl -X POST localhost:8000/validate -d '{"model":"llama-3.1-70b","tp":2,"pp":2,"dp":4}'

# 3. Generate commands
curl -X POST localhost:8000/launch -d '{"tp":2,"pp":2,"dp":4,"sharding":"zero3"}'
```

### Workflow 2: Cost Analysis
```bash
# 1. Estimate cost
curl -X POST localhost:8000/estimate -d '{"model":"llama-3.1-70b","tokens":1000000000000}'

# 2. Pareto analysis
curl -X POST localhost:8000/pareto -d '{"model":"llama-3.1-70b","gpu_cost":4.0}'
```

### Workflow 3: Inference Deployment
```bash
# 1. Inference optimization
curl -X POST localhost:8000/inference -d '{"model":"llama-3.1-70b","latency_target_ms":100}'

# 2. vLLM config
curl -X POST localhost:8000/vllm -d '{"model":"llama-3.1-70b","batch_size":64}'
```

---

**Need Help?**
- Interactive Docs: http://localhost:8000/docs
- Full Documentation: See `PARALLELISM_API_README.md`
- Examples: Run `python examples_api_usage.py`

