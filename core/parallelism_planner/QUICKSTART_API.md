# Parallelism Strategy Advisor API - Quick Start Guide

Get started with the Parallelism Strategy Advisor API in under 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerized deployment

## Installation

### 1. Install Dependencies

```bash
cd /path/to/Desero-pro

# Install required packages
pip install -r requirements.txt

# Install API-specific packages
pip install fastapi uvicorn[standard] pydantic
```

## Running the Server

### Option 1: Quick Start (Easiest)

```bash
# Start in development mode
./start_api.sh

# Or start directly with Python
python parallelism_planner_server.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Option 2: Using the Startup Script

```bash
# Development mode with auto-reload
./start_api.sh --dev

# Production mode with 4 workers
./start_api.sh --prod --workers 4

# Custom port
./start_api.sh --port 8080
```

### Option 3: Using uvicorn Directly

```bash
# Development mode
uvicorn parallelism_planner_server:app --reload --host 0.0.0.0 --port 8000

# Production mode with multiple workers
uvicorn parallelism_planner_server:app --workers 4 --host 0.0.0.0 --port 8000
```

### Option 4: Using Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Or build and run manually
docker build -f Dockerfile.api -t parallelism-api .
docker run -p 8000:8000 parallelism-api

# With nginx reverse proxy (production)
docker-compose --profile production up -d
```

## Testing the API

### 1. Quick Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### 2. Get API Information

```bash
curl http://localhost:8000/
```

### 3. Run Full Test Suite

```bash
python test_parallelism_api.py
```

This will test all API endpoints and show a detailed report.

## First API Call

### Example 1: Get Parallelism Recommendations

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

### Example 2: Generate Launch Commands

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
    "script": "train.py"
  }'
```

### Example 3: Get Model Analysis

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B",
    "batch_size": 8,
    "seq_length": 2048
  }'
```

## Using Python Client

```python
import requests

API_URL = "http://localhost:8000"

# Get recommendations
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
    print(result)
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## Interactive API Documentation

The FastAPI server provides interactive API documentation out of the box:

1. **Swagger UI**: http://localhost:8000/docs
   - Interactive interface to test all endpoints
   - Automatic request/response examples
   - Try out API calls directly from browser

2. **ReDoc**: http://localhost:8000/redoc
   - Beautiful, clean API documentation
   - Searchable endpoint list
   - Detailed request/response schemas

## Common Use Cases

### 1. Optimize Training Configuration

```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "training": true, "mock_topology": "h100"}'

# Validate configuration
curl -X POST "http://localhost:8000/validate" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "tp": 2, "pp": 2, "dp": 4, "memory": 80, "batch_size": 1, "seq_length": 2048}'

# Generate launch commands
curl -X POST "http://localhost:8000/launch" -H "Content-Type: application/json" \
  -d '{"nodes": 2, "gpus": 8, "tp": 2, "pp": 2, "dp": 4, "sharding": "zero3", "script": "train.py"}'
```

### 2. Optimize Inference Deployment

```bash
# Get inference optimization
curl -X POST "http://localhost:8000/inference" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "batch_size": 1, "seq_length": 2048, "max_output_length": 512, "latency_target_ms": 100}'

# Get vLLM configuration
curl -X POST "http://localhost:8000/vllm" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "batch_size": 64, "max_tokens": 512, "gpus": 2}'
```

### 3. Cost Analysis

```bash
# Estimate training cost
curl -X POST "http://localhost:8000/estimate" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "tokens": 1000000000000, "throughput": 100000, "gpus": 64, "gpu_cost": 4.0}'

# Pareto analysis
curl -X POST "http://localhost:8000/pareto" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "gpu_cost": 4.0, "batch_size": 8, "seq_length": 4096}'
```

### 4. Troubleshooting

```bash
# Get troubleshooting topics
curl -X POST "http://localhost:8000/troubleshoot" -H "Content-Type: application/json" -d '{}'

# Diagnose CUDA OOM error
curl -X POST "http://localhost:8000/troubleshoot" -H "Content-Type: application/json" \
  -d '{"error_message": "CUDA out of memory"}'

# Get memory breakdown
curl -X POST "http://localhost:8000/memory" -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-70B", "batch_size": 8, "seq_length": 2048, "tp": 2, "pp": 2, "training": true}'
```

## Production Deployment

### SystemD Service (Linux)

1. Edit the service file:
   ```bash
   sudo nano /etc/systemd/system/parallelism-api.service
   ```

2. Copy content from `parallelism-api.service` and update paths

3. Enable and start:
   ```bash
   sudo systemctl enable parallelism-api
   sudo systemctl start parallelism-api
   sudo systemctl status parallelism-api
   ```

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f parallelism-api

# Stop
docker-compose down
```

### Behind Nginx (Recommended for Production)

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d

# Or configure nginx manually using nginx.conf
sudo cp nginx.conf /etc/nginx/sites-available/parallelism-api
sudo ln -s /etc/nginx/sites-available/parallelism-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Troubleshooting

### API Won't Start

```bash
# Check if port is already in use
lsof -i :8000

# Use different port
./start_api.sh --port 8080
```

### Import Errors

```bash
# Verify all dependencies are installed
pip install -r requirements.txt
pip install fastapi uvicorn pydantic

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Connection Refused

```bash
# Check if server is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status
sudo ufw allow 8000
```

## Next Steps

- Read the full API documentation: [PARALLELISM_API_README.md](PARALLELISM_API_README.md)
- Explore interactive docs: http://localhost:8000/docs
- Run the test suite: `python test_parallelism_api.py`
- Check out example scripts in the documentation

## Support

For issues or questions:
- Check server logs
- Review error messages in API responses
- Consult the full documentation
- Test endpoints using the interactive docs at `/docs`

## Performance Tips

1. **Development**: Use `--reload` for auto-reload on code changes
2. **Production**: Use multiple workers (`--workers 4`)
3. **Load Balancing**: Use nginx reverse proxy for multiple instances
4. **Caching**: Consider adding Redis for result caching
5. **Rate Limiting**: Configure nginx rate limiting for production

Enjoy using the Parallelism Strategy Advisor API! 🚀

