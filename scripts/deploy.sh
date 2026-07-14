#!/usr/bin/env bash
# Deploy a new version of cortex to Cloud Run.
#
# Usage:
#   GCP_PROJECT=my-project ./scripts/deploy.sh               # deploy current HEAD
#   GCP_PROJECT=my-project ./scripts/deploy.sh --no-cache    # force full Docker rebuild

set -euo pipefail

PROJECT="${GCP_PROJECT:-}"
if [[ -z "$PROJECT" ]]; then
  echo "ERROR: GCP_PROJECT not set. Usage: GCP_PROJECT=my-project $0"
  exit 1
fi
REGION="${GCP_REGION:-us-central1}"
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

# Inject the SHA-tagged image (and project id) so Cloud Run sees a spec change
TMP_YAML=$(mktemp /tmp/service-XXXXXX.yaml)
sed "s|gcr.io/PROJECT_ID/cortex:latest|$TAG|g" "$SERVICE_YAML" > "$TMP_YAML"

if ! grep -q "$TAG" "$TMP_YAML"; then
  echo "ERROR: image placeholder gcr.io/PROJECT_ID/cortex:latest not found in $SERVICE_YAML"
  rm -f "$TMP_YAML"
  exit 1
fi

echo "▶ Deploying to Cloud Run..."
PREVIOUS_REVISION=$(gcloud run services describe cortex \
  --region="$REGION" \
  --project="$PROJECT" \
  --format="value(status.latestReadyRevisionName)" 2>/dev/null || true)
gcloud run services replace "$TMP_YAML" \
  --region="$REGION" \
  --project="$PROJECT"
rm -f "$TMP_YAML"

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

echo "▶ Running post-deployment smoke checks..."
if ! python3 scripts/post_deploy_smoke.py "$SERVICE_URL"; then
  echo "ERROR: deployment smoke checks failed; the new revision needs investigation."
  if [[ -n "$PREVIOUS_REVISION" ]]; then
    echo "Rollback: gcloud run services update-traffic cortex --region=$REGION --project=$PROJECT --to-revisions=$PREVIOUS_REVISION=100"
  else
    echo "No previous ready revision was found; inspect revisions before changing traffic."
  fi
  echo "Revisions: gcloud run revisions list --service=cortex --region=$REGION --project=$PROJECT"
  exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Deployed successfully"
echo "  URL    : $SERVICE_URL"
echo "  Health : $SERVICE_URL/health"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Next: verify Inngest serve URL is set to:"
echo "  $SERVICE_URL/api/inngest"
