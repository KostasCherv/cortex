#!/usr/bin/env bash
# Reset and rebuild the graphify knowledge graph (or incremental update).
#
# Usage:
#   ./scripts/graphify-rebuild.sh              # full reset + gemma4:31b-cloud
#   ./scripts/graphify-rebuild.sh --incremental
#   ./scripts/graphify-rebuild.sh --local      # local gemma4:e4b instead of cloud
#   ./scripts/graphify-rebuild.sh --no-cluster # skip GRAPH_REPORT.md / graph.html
#
# Requires: graphify CLI, Ollama running with the chosen model pulled.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/graphify-out"

MODEL="${GRAPHIFY_OLLAMA_MODEL:-gemma4:31b-cloud}"
INCREMENTAL=0
SKIP_CLUSTER=0

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --incremental) INCREMENTAL=1 ;;
    --local) MODEL="gemma4:e4b" ;;
    --model) MODEL="$2"; shift ;;
    --no-cluster) SKIP_CLUSTER=1 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

# Graphify uses the OpenAI-compatible API — base URL must end with /v1.
# (EMBEDDING_BASE_URL=http://localhost:11434 without /v1 will 404.)
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"

if [[ "${OLLAMA_BASE_URL}" != */v1 ]]; then
  echo "error: OLLAMA_BASE_URL must end with /v1 (got: ${OLLAMA_BASE_URL})" >&2
  echo "  fix: export OLLAMA_BASE_URL=http://localhost:11434/v1" >&2
  exit 1
fi

if ! command -v graphify >/dev/null 2>&1; then
  echo "error: graphify not found on PATH (try: uv tool install graphifyy)" >&2
  exit 1
fi

if ! curl -sf "${OLLAMA_BASE_URL%/v1}/api/tags" >/dev/null 2>&1; then
  echo "error: Ollama not reachable at ${OLLAMA_BASE_URL%/v1}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ "${INCREMENTAL}" -eq 0 ]]; then
  echo "[graphify-rebuild] full reset (semantic cache + graph outputs)"
  rm -f \
    "${OUT_DIR}/manifest.json" \
    "${OUT_DIR}/graph.json" \
    "${OUT_DIR}/.graphify_analysis.json" \
    "${OUT_DIR}/.graphify_labels.json"
  rm -rf "${OUT_DIR}/cache/semantic"
else
  echo "[graphify-rebuild] incremental (unchanged files use cache)"
fi

echo "[graphify-rebuild] model=${MODEL} backend=ollama"
graphify extract . \
  --backend ollama \
  --model "${MODEL}" \
  --max-concurrency 1 \
  --token-budget 12000 \
  --api-timeout 900

if [[ "${SKIP_CLUSTER}" -eq 0 ]]; then
  echo "[graphify-rebuild] cluster + report + html"
  graphify cluster-only . --backend=ollama
fi

if [[ -f "${OUT_DIR}/graph.json" ]]; then
  node_count="$(python3 -c "import json; print(len(json.load(open('${OUT_DIR}/graph.json'))['nodes']))")"
  echo "[graphify-rebuild] done: ${OUT_DIR}/graph.json (${node_count} nodes)"
  echo "  query: graphify query \"your question\""
else
  echo "error: graphify extract did not produce graph.json" >&2
  exit 1
fi
