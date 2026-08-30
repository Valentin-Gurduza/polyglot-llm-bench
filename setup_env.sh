#!/usr/bin/env bash
# =============================================================================
# Polyglot-LLM-Bench — One-Click Environment Setup
# =============================================================================
# Usage:  bash setup_env.sh
#
# Works on Linux, macOS, and Git Bash / WSL on Windows.
# Creates a Python virtual environment, installs dependencies, and prepares
# the .env file from the template.
# =============================================================================

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Locate project root (same directory as this script) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Project root: $SCRIPT_DIR"

# ── Detect Python ≥ 3.10 ────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.10+ is required but not found. Please install it first."
fi
ok "Found Python: $PYTHON ($($PYTHON --version 2>&1))"

# ── Create virtual environment ───────────────────────────────────────────────
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation."
else
    info "Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# ── Activate ─────────────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE="$VENV_DIR/Scripts/activate"
else
    ACTIVATE="$VENV_DIR/bin/activate"
fi

# shellcheck disable=SC1090
source "$ACTIVATE"
ok "Virtual environment activated."

# ── Upgrade pip & install dependencies ───────────────────────────────────────
info "Upgrading pip ..."
pip install --upgrade pip --quiet

info "Installing project dependencies ..."
pip install -r requirements.txt --quiet
ok "All dependencies installed."

# ── Prepare .env ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from .env.example — please edit it and add your OPENROUTER_API_KEY."
else
    ok ".env already exists — skipping copy."
fi

# ── Create output directories ────────────────────────────────────────────────
mkdir -p results .cache
touch results/.gitkeep .cache/.gitkeep

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅  Polyglot-LLM-Bench environment is ready!              ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Activate (bash/zsh): source .venv/bin/activate            ║${NC}"
echo -e "${GREEN}║  Activate (fish):     source .venv/bin/activate.fish       ║${NC}"
echo -e "${GREEN}║  Configure:           nano .env                            ║${NC}"
echo -e "${GREEN}║  Run:  python benchmark_runner.py run --dry-run            ║${NC}"
echo -e "${GREEN}║  Models: python benchmark_runner.py list-models            ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
