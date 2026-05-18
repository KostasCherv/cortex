#!/usr/bin/env bash
# Deploy a new version of cortex to Cloud Run.
#
# Usage:
#   ./scripts/deploy.sh               # deploy current HEAD
#   ./scripts/deploy.sh --no-cache    # force full Docker rebuild

set -euo pipefail

PROJECT="cortex-496709"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT/cortex"
SERVICE_YAML="cloudrun/service.yaml"

# ── Pre-flight ────────────────────────────────────────────────────────────────

command -v gcloud >/dev/null 2>&1 || { echo "ERROR: gcloud CLI not found."; exit 1; }

NO_CACHE=""
if [[ "${1:-}" == "--no-cache" ]]; then
  NO_CACHE="--no-cache"
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TAG="$IMAGE:$GIT_SHA"
LATEST="$IMAGE:latest"

echo "═══════════════════════════════════════════════════════"
echo "  Deploying cortex"
echo "  Commit : $GIT_SHA"
echo "  Image  : $TAG"
echo "  Region : $REGION"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Build & push ──────────────────────────────────────────────────────────────

echo "▶ Building image..."
gcloud builds submit \
  --tag "$TAG" \
  --project="$PROJECT" \
  $NO_CACHE \
  .

# Also tag as :latest so service.yaml always pulls the freshest build
echo "▶ Tagging as :latest..."
gcloud container images add-tag "$TAG" "$LATEST" --quiet

# ── Deploy ────────────────────────────────────────────────────────────────────

echo "▶ Deploying to Cloud Run..."
gcloud run services replace "$SERVICE_YAML" \
  --region="$REGION" \
  --project="$PROJECT"

# Ensure unauthenticated access (idempotent)
gcloud run services add-iam-policy-binding cortex \
  --region="$REGION" \
  --project="$PROJECT" \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --quiet 2>/dev/null || true

# ── Done ─────────────────────────────────────────────────────────────────────

SERVICE_URL=$(gcloud run services describe cortex \
  --region="$REGION" \
  --project="$PROJECT" \
  --format="value(status.url)")

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Deployed successfully"
echo "  URL    : $SERVICE_URL"
echo "  Health : $SERVICE_URL/health"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Next: verify Inngest serve URL is set to:"
echo "  $SERVICE_URL/api/inngest"
