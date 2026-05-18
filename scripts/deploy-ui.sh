#!/usr/bin/env bash
# Link and deploy cortex-ui to Vercel from the ui/ directory.
#
# Usage:
#   ./scripts/deploy-ui.sh              # preview deployment
#   ./scripts/deploy-ui.sh --prod       # production deployment
#   ./scripts/deploy-ui.sh --link-only  # first-time link, no deploy
#
# Prerequisites:
#   npx vercel login
#   Set production env vars (once): ./scripts/vercel-ui-env.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$ROOT_DIR/ui"
VERCEL=(npx --yes vercel@latest)

cd "$UI_DIR"

PROD=false
LINK_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --prod | prod) PROD=true ;;
    --link-only) LINK_ONLY=true ;;
  esac
done

if ! "${VERCEL[@]}" whoami >/dev/null 2>&1; then
  echo "ERROR: Not logged in to Vercel. Run: npx vercel login"
  exit 1
fi

echo "▶ Vercel account: $("${VERCEL[@]}" whoami)"

if [[ ! -f .vercel/project.json ]]; then
  echo "▶ Linking ui/ to a Vercel project (creates one if needed)..."
  "${VERCEL[@]}" link --yes
else
  echo "▶ Using linked project in ui/.vercel/project.json"
fi

if $LINK_ONLY; then
  echo "✓ Linked. Set env vars with ./scripts/vercel-ui-env.sh then deploy."
  exit 0
fi

echo "▶ Building locally..."
npm run build

if $PROD; then
  echo "▶ Deploying to production..."
  "${VERCEL[@]}" deploy --prod
else
  echo "▶ Deploying preview..."
  "${VERCEL[@]}" deploy
fi
