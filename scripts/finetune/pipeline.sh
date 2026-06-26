#!/usr/bin/env bash
# scripts/finetune/pipeline.sh
#
# Orchestrates the router fine-tuning pipeline around the GPU-only training step.
#
#   prepare   (local)  regenerate dataset via Ollama teacher + push to HF Hub
#   <manual>  (GPU)    run scripts/finetune/train_unsloth.ipynb on Kaggle/Colab;
#                      it pushes the LoRA adapter + GGUF to the Hub
#   activate  (local)  download the trained GGUF + ollama create + score
#
# Override any default via environment variables (see usage()).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo virtualenv if present so `python` resolves correctly.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ---- Config (override via env) ----
DATASET_REPO="${DATASET_REPO:-kostascherv/cortex-router-dataset}"
GGUF_REPO="${GGUF_REPO:-kostascherv/cortex-router-model-GGUF}"
GGUF_FILENAME="${GGUF_FILENAME:-router-model-q4_k_m.gguf}"
OLLAMA_MODEL="${OLLAMA_MODEL:-cortex-router}"

DATA_DIR="data/router_dataset"
TRAIN="$DATA_DIR/train.jsonl"
HELD_OUT="$DATA_DIR/held_out.jsonl"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# Resolve teacher model + backend + base URL through the same logic teacher_client uses.
resolve_env() {
  # Explicit path: find_dotenv() asserts on a missing __file__ when run via stdin.
  python - <<'PY'
import os
from dotenv import load_dotenv

load_dotenv(".env")
api = os.getenv("TEACHER_API", "ollama").strip().lower()
base = os.getenv("TEACHER_BASE_URL")
if not base:
    if api == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    else:
        base = "http://localhost:1234/v1"
print(os.getenv("TEACHER_MODEL", "qwen3:30b"))
print(api)
print(base.rstrip("/"))
PY
}

preflight_teacher() {
  local teacher api base
  { read -r teacher; read -r api; read -r base; } < <(resolve_env)
  log "Preflight: backend='$api' base='$base' teacher='$teacher'"
  if [[ "$api" == "openai" ]]; then
    # Resolve the key via command substitution so it is not echoed to the terminal.
    local key
    key="$(python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")
print(os.getenv("TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
PY
)"
    curl -fsS -H "Authorization: Bearer ${key}" "$base/models" >/dev/null 2>&1 \
      || die "OpenAI-compatible teacher not reachable/authorized at $base/models. For OpenAI set OPENAI_API_KEY; for LM Studio start its local server."
  else
    local tags
    tags="$(curl -fsS "$base/api/tags" 2>/dev/null)" \
      || die "Ollama not reachable at $base. Start it with 'ollama serve'."
    # Cloud models (e.g. *-cloud) are not listed in /api/tags, so this is a soft check.
    if ! grep -q "\"name\":\"$teacher\"" <<<"$tags" && [[ "$teacher" != *-cloud ]]; then
      warn "Teacher '$teacher' not found in local Ollama tags. Continuing (pull it if generation 404s)."
    fi
  fi
}

cmd_prepare() {
  preflight_teacher
  log "Regenerating dataset (scripts.generate_router_dataset)"
  python -m scripts.generate_router_dataset
  # Guard against the footgun: never push an empty/failed generation.
  [[ -s "$TRAIN" ]] || die "Generation produced empty $TRAIN — aborting before push (data backup retained)."
  [[ -s "$HELD_OUT" ]] || die "Generation produced empty $HELD_OUT — aborting before push (data backup retained)."
  log "Pushing dataset to HF Hub ($DATASET_REPO)"
  python -m scripts.finetune.push_to_hub --repo-id "$DATASET_REPO"
  log "PREPARE complete."
  cat <<EOF

  Next (manual, GPU): open scripts/finetune/train_unsloth.ipynb on Kaggle/Colab and run all cells.
  It pushes the LoRA adapter and GGUF to the Hub. When training finishes, run:

      $0 activate
EOF
}

cmd_activate() {
  log "Downloading trained GGUF ($GGUF_REPO/$GGUF_FILENAME)"
  hf download "$GGUF_REPO" "$GGUF_FILENAME" --local-dir scripts/finetune \
    || die "Download failed. Confirm training pushed '$GGUF_FILENAME' to $GGUF_REPO."
  [[ -f "scripts/finetune/$GGUF_FILENAME" ]] || die "GGUF not present after download."
  log "Registering with Ollama as '$OLLAMA_MODEL'"
  ollama create "$OLLAMA_MODEL" -f scripts/finetune/Modelfile
  cmd_score
}

cmd_score() {
  log "Scoring router on held-out set (scripts.score_router)"
  python -m scripts.score_router
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  prepare    Regenerate dataset (Ollama teacher) + push to HF Hub
  activate   Download trained GGUF from HF + ollama create + run scoring
  score      Run held-out scoring only
  help       Show this help

The training step between 'prepare' and 'activate' is GPU-only and runs
manually in scripts/finetune/train_unsloth.ipynb (Kaggle/Colab).

Defaults (override via env vars):
  DATASET_REPO   = $DATASET_REPO
  GGUF_REPO      = $GGUF_REPO
  GGUF_FILENAME  = $GGUF_FILENAME
  OLLAMA_MODEL   = $OLLAMA_MODEL
EOF
}

case "${1:-help}" in
  prepare) cmd_prepare ;;
  activate) cmd_activate ;;
  score) cmd_score ;;
  help | -h | --help) usage ;;
  *)
    usage
    die "Unknown command: ${1:-}"
    ;;
esac
