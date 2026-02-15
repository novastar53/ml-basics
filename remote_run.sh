#!/bin/bash
# remote_run.sh - Run Python scripts on remote GPU machines with log streaming
#
# Usage: ./remote_run.sh [-d] [-n SESSION] [-b BRANCH] <ssh-host> <script-path> [script-args...]
#        ./remote_run.sh <ssh-host> <command>
#
# Options:
#   -d           - Detached mode (run in background)
#   -n SESSION   - Custom tmux session name (default: ml_basics_remote)
#   -b BRANCH    - Git branch or commit to checkout before running
#
# Commands (when no script path provided):
#   attach       - Attach to running tmux session
#   stream       - Stream logs from remote to local
#   status       - Check if script is running
#   stop         - Stop script and download logs

set -e

# Configuration
REMOTE_DIR="~/ml-basics"
REPO_URL="https://github.com/novastar53/ml-basics.git"
DEFAULT_SESSION="ml_basics_remote"
LOG_DIR="remote_logs"
REMOTE_LOG_DIR="~/.cache/ml_basics_remote"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse options
DETACH=false
SESSION_NAME="$DEFAULT_SESSION"
GIT_REF=""

while getopts "dn:b:" opt; do
    case $opt in
        d)
            DETACH=true
            ;;
        n)
            SESSION_NAME="$OPTARG"
            ;;
        b)
            GIT_REF="$OPTARG"
            ;;
        \?)
            echo -e "${RED}Invalid option: -$OPTARG${NC}" &&2
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# Parse positional arguments
SSH_HOST=${1:-}
SECOND_ARG=${2:-}

if [ -z "$SSH_HOST" ]; then
    echo -e "${RED}Error: SSH host is required${NC}"
    exit 1
fi

# Detect if second argument is a command or script path
COMMANDS="attach|stream|status|stop"
if [[ "$SECOND_ARG" =~ ^($COMMANDS)$ ]]; then
    COMMAND="$SECOND_ARG"
    SCRIPT_PATH=""
    SCRIPT_ARGS=""
else
    COMMAND="run"
    SCRIPT_PATH="$SECOND_ARG"
    shift 2
    SCRIPT_ARGS="$@"
fi

# Generate timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Helper function to run commands on remote
remote_exec() {
    ssh "$SSH_HOST" "$@"
}

# Setup remote environment
setup_remote() {
    echo -e "${GREEN}Setting up remote environment...${NC}"

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

REMOTE_DIR_EXPANDED=\$(eval echo ~/ml-basics)
PARENT_DIR=\$(dirname "\$REMOTE_DIR_EXPANDED")

# Clone or update repo
if [ -d "\$REMOTE_DIR_EXPANDED" ]; then
    echo "Updating existing repo..."
    cd "\$REMOTE_DIR_EXPANDED"
    git fetch origin
    git checkout main || true
    git pull origin main
else
    echo "Cloning repo..."
    mkdir -p "\$PARENT_DIR"
    git clone $REPO_URL "\$REMOTE_DIR_EXPANDED"
    cd "\$REMOTE_DIR_EXPANDED"
fi

# Checkout specified branch/commit if provided
GIT_REF="$GIT_REF"
if [ -n "\$GIT_REF" ]; then
    echo "Checking out: \$GIT_REF"
    git checkout "\$GIT_REF"
fi

# Clone or update jax-flow dependency (required sibling directory)
JAXFLOW_DIR="\$PARENT_DIR/jax_fusion"
if [ -d "\$JAXFLOW_DIR" ]; then
    echo "Updating jax-flow dependency..."
    cd "\$JAXFLOW_DIR"
    git fetch origin
    git pull origin main || true
else
    echo "Cloning jax-flow dependency..."
    mkdir -p "\$PARENT_DIR"
    # Try jax-flow first, fall back to jax_fusion
    git clone https://github.com/novastar53/jax-flow "\$JAXFLOW_DIR" 2>/dev/null || \
        git clone https://github.com/novastar53/jax_fusion "\$JAXFLOW_DIR"
fi

# Create log directory
mkdir -p ~/.cache/ml_basics_remote

# Source uv environment
if [ -f "\$HOME/.local/bin/env" ]; then
    export TERM=xterm-256color
    source "\$HOME/.local/bin/env"
fi

# Install tmux if needed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y tmux
    elif command -v yum &> /dev/null; then
        yum install -y tmux
    elif command -v apk &> /dev/null; then
        apk add tmux
    fi
fi

# Install dependencies if needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "\$HOME/.local/bin/env"
fi

# Install project dependencies
echo "Ensuring dependencies are installed..."
cd "\$REMOTE_DIR_EXPANDED"
uv sync --extra gpu 2>/dev/null || uv sync

echo "Remote setup complete!"
REMOTE_SCRIPT
}

run_script_foreground() {
    local script_path="$1"
    local script_args="$2"
    local log_file="$REMOTE_LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"

    echo -e "${GREEN}Running script on $SSH_HOST (foreground mode)...${NC}"

    mkdir -p "$LOG_DIR"
    local local_log="$LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"

    remote_exec "tmux kill-session -t $SESSION_NAME 2>/dev/null" || true

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)
cd "\$REMOTE_DIR_EXPANDED"

source \$HOME/.local/bin/env 2>/dev/null

echo "Starting: uv run python $script_path $script_args"
uv run python $script_path $script_args 2>&1 | tee $log_file
REMOTE_SCRIPT

    echo ""
    echo -e "${GREEN}Script completed. Downloading logs...${NC}"
    remote_exec "cat $log_file" > "$local_log"
    echo -e "${GREEN}Logs saved to: $local_log${NC}"
}

run_script_background() {
    local script_path="$1"
    local script_args="$2"
    local log_file="$REMOTE_LOG_DIR/${SESSION_NAME}_${TIMESTAMP}.log"

    echo -e "${GREEN}Running script on $SSH_HOST (background mode)...${NC}"
    echo -e "${BLUE}Session: $SESSION_NAME${NC}"

    remote_exec "tmux kill-session -t $SESSION_NAME 2>/dev/null" || true

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)
cd "\$REMOTE_DIR_EXPANDED"

source \$HOME/.local/bin/env 2>/dev/null

tmux kill-session -t $SESSION_NAME 2>/dev/null || true
tmux new-session -d -s $SESSION_NAME -c "\$REMOTE_DIR_EXPANDED"
tmux send-keys -t $SESSION_NAME "source \$HOME/.local/bin/env 2>/dev/null" Enter
tmux send-keys -t $SESSION_NAME "uv run python $script_path $script_args 2>&1 | tee $log_file" Enter

echo "Script started in tmux session '$SESSION_NAME'"
REMOTE_SCRIPT

    echo -e "${GREEN}Script started in background.${NC}"
    echo -e "${YELLOW}Use './remote_run.sh $SSH_HOST attach' to attach${NC}"
    echo -e "${YELLOW}Use './remote_run.sh $SSH_HOST stream' to stream logs${NC}"
}

cmd_run() {
    if [ -z "$SCRIPT_PATH" ]; then
        echo -e "${RED}Error: Script path is required${NC}"
        exit 1
    fi

    setup_remote

    if [ "$DETACH" = true ]; then
        run_script_background "$SCRIPT_PATH" "$SCRIPT_ARGS"
    else
        run_script_foreground "$SCRIPT_PATH" "$SCRIPT_ARGS"
    fi
}

cmd_attach() {
    echo -e "${GREEN}Attaching to tmux session '$SESSION_NAME' on $SSH_HOST...${NC}"
    ssh -t "$SSH_HOST" "tmux attach-session -t $SESSION_NAME"
}

cmd_stream() {
    echo -e "${GREEN}Streaming logs from $SSH_HOST...${NC}"
    mkdir -p "$LOG_DIR"

    local remote_log=$(remote_exec "ls -t ~/.cache/ml_basics_remote/${SESSION_NAME}_*.log 2>/dev/null | head -1" || echo "")

    if [ -z "$remote_log" ]; then
        echo -e "${RED}No log files found${NC}"
        exit 1
    fi

    ssh "$SSH_HOST" "tail -f $remote_log"
}

cmd_status() {
    echo -e "${GREEN}Checking status on $SSH_HOST...${NC}"

    remote_exec "bash -l" << 'REMOTE_SCRIPT'
if tmux has-session -t SESSION_NAME 2>/dev/null; then
    echo "Tmux session: RUNNING"
else
    echo "Tmux session: NOT RUNNING"
fi

if pgrep -f "uv run python" > /dev/null; then
    echo "Python process: RUNNING"
else
    echo "Python process: NOT RUNNING"
fi
REMOTE_SCRIPT
}

cmd_stop() {
    echo -e "${GREEN}Stopping script on $SSH_HOST...${NC}"

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "No tmux session found."
    exit 0
fi

tmux send-keys -t $SESSION_NAME C-c
sleep 2
pkill -9 -f "uv run python" 2>/dev/null || true
tmux kill-session -t $SESSION_NAME 2>/dev/null || true
echo "Stopped."
REMOTE_SCRIPT
}

# Execute command
case "$COMMAND" in
    run)
        cmd_run
        ;;
    attach)
        cmd_attach
        ;;
    stream)
        cmd_stream
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        exit 1
        ;;
esac
