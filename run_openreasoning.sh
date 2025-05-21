#!/bin/bash
# Run OpenReasoning

# Terminal colors for better readability
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Options
ENV_FILE=".env"
MODE="cli"
PORT=8000
HOST="localhost"

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --api)
            MODE="api"
            shift
            ;;
        --port)
            PORT="$2"
            shift
            shift
            ;;
        --host)
            HOST="$2"
            shift
            shift
            ;;
        --env)
            ENV_FILE="$2"
            shift
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if running on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    # Check if running on Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo -e "${MAGENTA}Running on Apple Silicon Mac${NC}"

        # Check if running on M3 series
        M3_CHIP=$(sysctl -n machdep.cpu.brand_string | grep -i "M3")
        if [[ ! -z "$M3_CHIP" ]]; then
            echo -e "${MAGENTA}M3 series chip detected: $M3_CHIP${NC}"
            echo -e "${GREEN}M3 optimizations will be automatically applied${NC}"
        fi
    fi
fi

# Check if env file exists
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}Loading environment from $ENV_FILE${NC}"
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo -e "${YELLOW}No .env file found at $ENV_FILE. Using existing environment variables.${NC}"
fi

# Check for required API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}OPENAI_API_KEY is not set. Some functionality may be limited.${NC}"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Run in the selected mode
if [ "$MODE" = "api" ]; then
    echo -e "${BLUE}Starting OpenReasoning API server on $HOST:$PORT${NC}"
    python -m openreasoning.cli server --host "$HOST" --port "$PORT"
else
    echo -e "${BLUE}Starting OpenReasoning CLI${NC}"
    python -m openreasoning.cli
fi 