#!/usr/bin/env bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tscircuit/dataset-zero-obstacle-high-density-z01.git}"
REPO_DIR="${REPO_DIR:-dataset-zero-obstacle-high-density-z01}"
TMUX_SESSION="${TMUX_SESSION:-dataset-generate}"

echo "[1/5] Installing Bun (if needed)"
if ! command -v bun >/dev/null 2>&1; then
  curl -fsSL https://bun.sh/install | bash
fi

echo "[2/5] Configuring Bun PATH"

export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$BUN_INSTALL/bin:$PATH"

if ! command -v bun >/dev/null 2>&1; then
  echo "Error: Bun is still not available on PATH after installation." >&2
  exit 1
fi

echo "[3/5] Cloning repo (if needed)"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[4/5] Installing repo dependencies"
cd "$REPO_DIR"
bun install

echo "[5/5] Starting dataset generation in detached tmux"
if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed. Install tmux and rerun this script." >&2
  exit 1
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  tmux kill-session -t "$TMUX_SESSION"
fi

tmux new-session -d -s "$TMUX_SESSION" "bun run scripts/generate-dataset-parallel.ts --sample-count 1000000 --output-dir dataset-256 --concurrency 32"

echo "Done. Session '$TMUX_SESSION' started."
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Tail logs: tmux capture-pane -p -t $TMUX_SESSION"
