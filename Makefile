.PHONY: install clean sync help

# Default SSH host (from ~/.ssh/config)
host ?= gpu

# Detect platform
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

# Set JAX extras based on platform
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        JAX_PLATFORM = metal
    else
        JAX_PLATFORM = cpu
    endif
else ifeq ($(UNAME_S),Linux)
    ifeq ($(shell command -v nvidia-smi > /dev/null 2>&1 && echo yes),yes)
        JAX_PLATFORM = gpu
    else
        JAX_PLATFORM = cpu
    endif
else
    JAX_PLATFORM = cpu
endif

print-platform:
	@echo "JAX_PLATFORM: $(JAX_PLATFORM)"

# Install dependencies with platform-specific JAX
install:
	@command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv sync --extra $(JAX_PLATFORM)

# Regenerate lockfile from scratch
regen-lock:
	rm -f uv.lock
	uv sync --extra $(JAX_PLATFORM)

# Add a production dependency (usage: make add pkg=package_name)
add:
	uv add $(pkg)

# Remove a dependency (usage: make remove pkg=package_name)
remove:
	uv remove $(pkg)

# Clean build artifacts and cache
clean:
	rm -f uv.lock
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf **/__pycache__/
	rm -rf .venv/

# Show installed packages
list:
	uv pip list

# Check JAX installation and GPU availability
verify-jax:
	uv run python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

# Help command
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies with platform-specific JAX"
	@echo "  make regen-lock    - Regenerate lockfile from scratch"
	@echo "  make add           - Add a dependency (make add pkg=package_name)"
	@echo "  make remove        - Remove a dependency (make remove pkg=package_name)"
	@echo "  make clean         - Clean build artifacts and cache"
	@echo "  make list          - Show installed packages"
	@echo "  make verify-jax    - Check JAX installation and devices"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Remote Execution (via remote_run.sh):"
	@echo "  ./remote_run.sh <host> <script-path> [args]  - Run script on remote machine"
	@echo "  ./remote_run.sh -d <host> <script>           - Run in background"
	@echo "  ./remote_run.sh <host> attach                - Attach to session"
	@echo "  ./remote_run.sh <host> status                - Check status"
	@echo "  ./remote_run.sh <host> stop                  - Stop and download logs"
