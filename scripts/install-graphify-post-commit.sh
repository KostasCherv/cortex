#!/usr/bin/env bash
# Wire scripts/graphify-post-commit.sh into .git/hooks/post-commit.
# Safe to re-run after `graphify hook install` (re-applies the research_agent block).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK="${ROOT}/.git/hooks/post-commit"
DRIVER="${ROOT}/scripts/graphify-post-commit.sh"

chmod +x "${DRIVER}"

if [[ ! -f "${HOOK}" ]]; then
  echo "error: ${HOOK} not found. Run: graphify hook install" >&2
  exit 1
fi

# Replace graphify's inline nohup python with our driver (inside graphify-hook block).
python3 <<'PY'
import re
from pathlib import Path

hook = Path(".git/hooks/post-commit")
text = hook.read_text(encoding="utf-8")
root = Path(".").resolve()
driver = root / "scripts/graphify-post-commit.sh"

replacement = f'''export GRAPHIFY_CHANGED="$CHANGED"

# Run rebuild detached so git commit returns immediately.
_GRAPHIFY_LOG="${{HOME}}/.cache/graphify-rebuild.log"
mkdir -p "$(dirname "$_GRAPHIFY_LOG")"
echo "[graphify hook] launching background rebuild (log: $_GRAPHIFY_LOG)"
nohup "{driver}" >> "$_GRAPHIFY_LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true'''

pattern = re.compile(
    r"export GRAPHIFY_CHANGED=\"\$CHANGED\"\n\n"
    r"# Run rebuild detached.*?disown 2>/dev/null \|\| true",
    re.DOTALL,
)
if not pattern.search(text):
    print("error: could not find graphify nohup block to patch", flush=True)
    raise SystemExit(1)
text = pattern.sub(replacement, text, count=1)
hook.write_text(text, encoding="utf-8")
print("patched graphify nohup -> scripts/graphify-post-commit.sh")
PY

echo "Done. On .md/.mdx commits: incremental Ollama extract runs in background."
echo "Log: tail -f ~/.cache/graphify-rebuild.log"
echo "Re-run this script after: graphify hook install"
