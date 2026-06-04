#!/usr/bin/env bash
# Post-commit graphify driver (research_agent).
# - Code-only commits: AST rebuild via graphify (no LLM).
# - Doc/image commits (.md, .mdx, …): incremental graphify extract (Ollama).
# - Mixed commits: incremental only (avoids two writers racing on graph.json).
#
# Invoked from .git/hooks/post-commit (see scripts/install-graphify-post-commit.sh).

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${ROOT_DIR}"

LOG="${HOME}/.cache/graphify-rebuild.log"
mkdir -p "$(dirname "${LOG}")"

CHANGED_RAW="${GRAPHIFY_CHANGED:-}"
if [[ -z "${CHANGED_RAW}" ]]; then
  CHANGED_RAW="$(git diff --name-only HEAD~1 HEAD 2>/dev/null || git diff --name-only HEAD 2>/dev/null || true)"
fi
[[ -n "${CHANGED_RAW}" ]] || exit 0

has_semantic=0
has_code=0
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  case "${f##*.}" in
    [Mm][Dd]|[Mm][Dd][Xx]|[Rr][Ss][Tt]|[Tt][Xx][Tt]|[Pp][Dd][Ff]|[Pp][Nn][Gg]|[Jj][Pp][Gg]|[Jj][Pp][Ee][Gg]|[Gg][Ii][Ff]|[Ww][Ee][Bb][Pp]|[Ss][Vv][Gg])
      has_semantic=1
      ;;
    [Pp][Yy]|[Pp][Yy][Ii]|[Tt][Ss]|[Tt][Ss][Xx]|[Jj][Ss]|[Jj][Ss][Xx]|[Gg][Oo]|[Rr][Ss]|[Jj][Aa][Vv][Aa]|[Rr][Bb]|[Cc]|[Hh]|[Cc][Pp][Pp]|[Hh][Pp][Pp]|[Cc][Cc]|[Cc][Ss]|[Kk][Tt]|[Ss][Ww][Ii][Ff][Tt]|[Pp][Hh][Pp]|[Ss][Cc][Aa][Ll][Aa]|[Ll][Uu][Aa]|[Ss][Hh]|[Ss][Qq][Ll]|[Zz][Ii][Gg]|[Vv][Uu][Ee]|[Ss][Vv][Ee][Ll][Tt][Ee]|[Aa][Ss][Tt][Rr][Oo])
      has_code=1
      ;;
  esac
done <<< "${CHANGED_RAW}"

export PYTHONHASHSEED=0
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-ollama}"
MODEL="${GRAPHIFY_OLLAMA_MODEL:-gemma4:31b-cloud}"

if [[ "${OLLAMA_BASE_URL}" != */v1 ]]; then
  echo "[graphify post-commit] error: OLLAMA_BASE_URL must end with /v1 (got ${OLLAMA_BASE_URL})" >> "${LOG}"
  exit 1
fi

GRAPH_JSON="${ROOT_DIR}/graphify-out/graph.json"

if [[ "${has_semantic}" -eq 1 ]]; then
  if [[ ! -f "${GRAPH_JSON}" ]]; then
    echo "[graphify post-commit] skip semantic: no graph.json yet (run ./scripts/graphify-rebuild.sh first)" >> "${LOG}"
    exit 0
  fi
  echo "[graphify post-commit] semantic/doc change -> incremental extract (model=${MODEL})" >> "${LOG}"
  if ! command -v graphify >/dev/null 2>&1; then
    echo "[graphify post-commit] error: graphify not on PATH" >> "${LOG}"
    exit 1
  fi
  graphify extract . \
    --backend ollama \
    --model "${MODEL}" \
    --max-concurrency 1 \
    --token-budget 12000 \
    --api-timeout 900 >> "${LOG}" 2>&1
  graphify cluster-only . --backend=ollama >> "${LOG}" 2>&1 || true
  echo "[graphify post-commit] incremental semantic done" >> "${LOG}"
  exit 0
fi

if [[ "${has_code}" -eq 1 ]]; then
  echo "[graphify post-commit] code change -> AST rebuild" >> "${LOG}"
  PYTHON=""
  _PINNED='/Users/kostas/.local/pipx/venvs/graphifyy/bin/python'
  if [[ -x "${_PINNED}" ]] && "${_PINNED}" -c "import graphify" 2>/dev/null; then
    PYTHON="${_PINNED}"
  elif [[ -f graphify-out/.graphify_python ]]; then
    PYTHON="$(tr -d '[:space:]' < graphify-out/.graphify_python)"
  elif command -v graphify >/dev/null 2>&1; then
    _bin="$(command -v graphify)"
    PYTHON="$(head -1 "${_bin}" | sed 's/^#!//;s/^[[:space:]]*//')"
    case "${PYTHON}" in */env\ *) PYTHON="${PYTHON#*/env }" ;; esac
  else
    PYTHON="python3"
  fi

  export GRAPHIFY_CHANGED="${CHANGED_RAW}"
  "${PYTHON}" -c "
import os, signal, sys
from pathlib import Path

changed_raw = os.environ.get('GRAPHIFY_CHANGED', '')
changed = [Path(f.strip()) for f in changed_raw.strip().splitlines() if f.strip()]
if not changed:
    sys.exit(0)
from graphify.watch import _rebuild_code, _apply_resource_limits
_apply_resource_limits()
_timeout = int(os.environ.get('GRAPHIFY_REBUILD_TIMEOUT', '600'))
if _timeout > 0 and hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(f'graphify rebuild exceeded {_timeout}s')))
    signal.alarm(_timeout)
_force = os.environ.get('GRAPHIFY_FORCE', '').lower() in ('1', 'true', 'yes')
_rebuild_code(Path('.'), changed_paths=changed, force=_force)
" >> "${LOG}" 2>&1
  echo "[graphify post-commit] AST rebuild done" >> "${LOG}"
fi
