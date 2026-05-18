#!/usr/bin/env bash
# Push VITE_* variables from .env.prod into Vercel (production + preview).
#
# Usage:
#   ./scripts/vercel-ui-env.sh
#
# Requires: ui/.vercel/project.json (run ./scripts/deploy-ui.sh --link-only first)
# Reads:    .env.prod at repo root (VITE_* keys only)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$ROOT_DIR/ui"
ENV_FILE="$ROOT_DIR/.env.prod"
UI_ENV_FILE="$UI_DIR/.env.prod"
VERCEL=(npx --yes vercel@latest)

cd "$UI_DIR"

if [[ ! -f .vercel/project.json ]]; then
  echo "ERROR: ui/ is not linked. Run: ./scripts/deploy-ui.sh --link-only"
  exit 1
fi

if [[ ! -f "$ENV_FILE" && ! -f "$UI_ENV_FILE" ]]; then
  echo "ERROR: Missing $ENV_FILE or $UI_ENV_FILE"
  exit 1
fi

if ! "${VERCEL[@]}" whoami >/dev/null 2>&1; then
  echo "ERROR: Not logged in. Run: npx vercel login"
  exit 1
fi

set -a
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
if [[ -f "$UI_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$UI_ENV_FILE"
fi
set +a

push_env() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "WARN: $name is empty in .env.prod — skipping"
    return
  fi
  for target in production preview; do
    "${VERCEL[@]}" env add "$name" "$target" --value "$value" --force --yes
    echo "  ✓ $name → $target"
  done
}

echo "▶ Syncing VITE_* env vars to Vercel..."
push_env VITE_API_BASE_URL
push_env VITE_SUPABASE_URL
push_env VITE_SUPABASE_PUBLISHABLE_KEY
echo "✓ Done. Redeploy with ./scripts/deploy-ui.sh --prod"
