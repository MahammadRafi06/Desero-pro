#!/bin/bash
#
# Start script for Parallelism Strategy Advisor API
#
# Usage:
#   ./start_api.sh [options]
#
# Options:
#   --dev       Start in development mode with auto-reload
#   --prod      Start in production mode with multiple workers
#   --port      Specify port (default: 8000)
#   --workers   Number of workers for production mode (default: 4)
#   --help      Show this help message
#

set -e

# Default values
MODE="dev"
PORT=8000
WORKERS=4
HOST="0.0.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dev       Start in development mode with auto-reload (default)"
            echo "  --prod      Start in production mode with multiple workers"
            echo "  --port      Specify port (default: 8000)"
            echo "  --workers   Number of workers for production mode (default: 4)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Parallelism Strategy Advisor API${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import fastapi, uvicorn, pydantic" 2>/dev/null || {
    echo -e "${RED}Error: Required packages not installed${NC}"
    echo "Please run: pip install -r requirements.txt"
    echo "And: pip install fastapi uvicorn pydantic"
    exit 1
}
echo -e "${GREEN}✓ Dependencies OK${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Host: $HOST"
echo "  Port: $PORT"
if [ "$MODE" == "prod" ]; then
    echo "  Workers: $WORKERS"
fi
echo "  Directory: $SCRIPT_DIR"
echo ""

# Start server based on mode
if [ "$MODE" == "dev" ]; then
    echo -e "${GREEN}Starting in DEVELOPMENT mode...${NC}"
    echo -e "${YELLOW}API will be available at: http://localhost:$PORT${NC}"
    echo -e "${YELLOW}Documentation: http://localhost:$PORT/docs${NC}"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    uvicorn parallelism_planner_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level info

elif [ "$MODE" == "prod" ]; then
    echo -e "${GREEN}Starting in PRODUCTION mode...${NC}"
    echo -e "${YELLOW}API will be available at: http://localhost:$PORT${NC}"
    echo -e "${YELLOW}Documentation: http://localhost:$PORT/docs${NC}"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    uvicorn parallelism_planner_server:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info \
        --access-log

else
    echo -e "${RED}Invalid mode: $MODE${NC}"
    exit 1
fi

