#!/usr/bin/env bash
# First-time setup: creates all required secrets in Google Secret Manager.
# Run once before the first deploy.
#
# Usage: ./scripts/setup_secrets.sh
# Then populate each secret:
#   echo -n "sk-..." | gcloud secrets versions add openai-api-key --data-file=-

set -euo pipefail

PROJECT="cortex-496709"
REGION="us-central1"

# ── Pre-flight ────────────────────────────────────────────────────────────────

command -v gcloud >/dev/null 2>&1 || { echo "ERROR: gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"; exit 1; }

echo "Configuring project: $PROJECT"
gcloud config set project "$PROJECT" --quiet

echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  --project="$PROJECT" \
  --quiet

# ── Create secrets (idempotent) ───────────────────────────────────────────────

secrets=(
  openai-api-key
  tavily-api-key
  cohere-api-key
  neo4j-uri
  neo4j-username
  neo4j-password
  supabase-secret-key
  supabase-jwt-secret
  inngest-event-key
  inngest-signing-key
  redis-url
  stripe-secret-key
  stripe-webhook-secret
  internal-dispatch-secret
  langfuse-public-key
  langfuse-secret-key
)

echo ""
echo "Creating secrets..."
for secret in "${secrets[@]}"; do
  if gcloud secrets describe "$secret" --project="$PROJECT" >/dev/null 2>&1; then
    echo "  ✓ $secret (already exists)"
  else
    echo -n "PLACEHOLDER" | gcloud secrets create "$secret" \
      --project="$PROJECT" \
      --data-file=- \
      --replication-policy=automatic \
      --quiet
    echo "  + $secret (created)"
  fi
done

# ── Generate internal dispatch secret ────────────────────────────────────────

echo ""
echo "Generating internal-dispatch-secret value..."
DISPATCH_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
echo -n "$DISPATCH_SECRET" | gcloud secrets versions add internal-dispatch-secret \
  --project="$PROJECT" \
  --data-file=- \
  --quiet
echo "  ✓ internal-dispatch-secret set"

# ── Instructions ─────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Secrets created. Now populate each one:"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  echo -n 'VALUE' | gcloud secrets versions add SECRET_NAME --data-file=-"
echo ""
echo "  Secrets to populate:"
remaining=(
  openai-api-key
  tavily-api-key
  cohere-api-key
  neo4j-uri
  neo4j-username
  neo4j-password
  supabase-secret-key
  supabase-jwt-secret
  inngest-event-key
  inngest-signing-key
  redis-url
  stripe-secret-key
  stripe-webhook-secret
  langfuse-public-key
  langfuse-secret-key
)
for s in "${remaining[@]}"; do
  echo "    - $s"
done
echo ""
echo "  See docs/env-vars-production.md for full reference."
echo ""
echo "  Once secrets are populated, run: ./scripts/deploy.sh"
