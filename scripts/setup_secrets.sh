#!/usr/bin/env bash
# First-time setup: creates the required secrets in Google Secret Manager.
# Run once before the first deploy.
#
# Usage: GCP_PROJECT=my-project ./scripts/setup_secrets.sh
#    or: ./scripts/setup_secrets.sh my-project
#
# If .env.prod exists at the repo root, secret values are populated from it
# automatically. Otherwise populate each secret manually.
#
# NOTE: the set below is 6 secrets, matching Secret Manager's free active-version
# allowance. Related credentials are stored as JSON configuration bundles. When rotating a value,
# destroy the old version to avoid accumulating billable versions:
#   gcloud secrets versions destroy <N> --secret=<name>

set -euo pipefail

PROJECT="${GCP_PROJECT:-${1:-}}"
if [[ -z "$PROJECT" ]]; then
  echo "ERROR: no project set. Usage: GCP_PROJECT=my-project $0  (or: $0 my-project)"
  exit 1
fi
REGION="${GCP_REGION:-us-central1}"

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
# Related credentials share JSON configuration secrets to stay within the six
# free active versions. Non-sensitive config remains in cloudrun/service.yaml.

secrets=(
  provider-config
  neo4j-password
  supabase-secret-key
  inngest-config
  billing-config
  internal-dispatch-secret
)

echo ""
echo "Creating secrets..."
for secret in "${secrets[@]}"; do
  if gcloud secrets describe "$secret" --project="$PROJECT" >/dev/null 2>&1; then
    echo "  ✓ $secret (already exists)"
  else
    # Created empty (no version) so the real value is version 1 — versions bill.
    gcloud secrets create "$secret" \
      --project="$PROJECT" \
      --replication-policy=automatic \
      --quiet
    echo "  + $secret (created)"
  fi
done

# ── Populate values from .env.prod (if present) ──────────────────────────────

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env.prod"

if [[ -f "$ENV_FILE" ]]; then
  echo ""
  echo "Populating secret values from .env.prod..."
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  populate() {
    local secret="$1" var="$2"
    local value="${!var:-}"
    if [[ -z "$value" ]]; then
      echo "  ! $secret ($var empty in .env.prod — populate manually)"
      return
    fi
    echo -n "$value" | gcloud secrets versions add "$secret" \
      --project="$PROJECT" --data-file=- --quiet >/dev/null
    echo "  ✓ $secret"
  }

  populate_bundle() {
    local secret="$1"
    shift
    local payload
    payload="$(python3 - "$@" <<'PY'
import json
import os
import sys

values = {}
for mapping in sys.argv[1:]:
    field, env_var = mapping.split("=", 1)
    value = os.environ.get(env_var, "")
    if value:
        values[field] = value
print(json.dumps(values, separators=(",", ":")))
PY
)"
    echo -n "$payload" | gcloud secrets versions add "$secret" \
      --project="$PROJECT" --data-file=- --quiet >/dev/null
    echo "  ✓ $secret"
  }

  if [[ -n "${OPENAI_API_KEY:-}" && -n "${TAVILY_API_KEY:-}" ]]; then
    populate_bundle provider-config \
      openai_api_key=OPENAI_API_KEY \
      tavily_api_key=TAVILY_API_KEY \
      redis_url=REDIS_URL \
      langfuse_public_key=LANGFUSE_PUBLIC_KEY \
      langfuse_secret_key=LANGFUSE_SECRET_KEY \
      langfuse_base_url=LANGFUSE_HOST \
      langsmith_api_key=LANGSMITH_API_KEY \
      langsmith_project=LANGSMITH_PROJECT \
      langsmith_endpoint=LANGSMITH_ENDPOINT \
      langsmith_redaction_mode=LANGSMITH_REDACTION_MODE \
      langsmith_sampling_rate=LANGSMITH_SAMPLING_RATE \
      langsmith_tracing=LANGSMITH_TRACING \
      sentry_dsn=SENTRY_DSN
  else
    echo "  ! provider-config (OPENAI_API_KEY or TAVILY_API_KEY empty — populate manually)"
  fi

  populate neo4j-password NEO4J_PASSWORD
  populate supabase-secret-key SUPABASE_SECRET_KEY

  if [[ -n "${INNGEST_EVENT_KEY:-}" && -n "${INNGEST_SIGNING_KEY:-}" ]]; then
    populate_bundle inngest-config \
      inngest_event_key=INNGEST_EVENT_KEY \
      inngest_signing_key=INNGEST_SIGNING_KEY
  else
    echo "  ! inngest-config (INNGEST_EVENT_KEY or INNGEST_SIGNING_KEY empty — populate manually)"
  fi

  if [[ -n "${STRIPE_SECRET_KEY:-}" && -n "${STRIPE_WEBHOOK_SECRET:-}" && -n "${STRIPE_PRO_PRICE_ID:-}" ]]; then
    populate_bundle billing-config \
      stripe_secret_key=STRIPE_SECRET_KEY \
      stripe_webhook_secret=STRIPE_WEBHOOK_SECRET \
      stripe_pro_price_id=STRIPE_PRO_PRICE_ID
  else
    echo "  ! billing-config (one or more Stripe values empty — populate manually)"
  fi
else
  echo ""
  echo "No .env.prod found — populate secrets manually (see instructions below)."
fi

# ── Grant Cloud Run access to secrets ────────────────────────────────────────
# Cloud Run runs as the default compute service account; it needs read access
# to the secrets referenced in cloudrun/service.yaml.

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo ""
echo "Granting secretAccessor to $COMPUTE_SA..."
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:$COMPUTE_SA" \
  --role="roles/secretmanager.secretAccessor" \
  --condition=None \
  --quiet >/dev/null
echo "  ✓ done"

# ── Generate internal dispatch secret ────────────────────────────────────────

echo ""
echo "Checking internal-dispatch-secret value..."
ACTIVE_DISPATCH_VERSION=$(gcloud secrets versions list internal-dispatch-secret \
  --project="$PROJECT" \
  --filter="state=ENABLED" \
  --format="value(name)" \
  --limit=1)
if [[ -n "$ACTIVE_DISPATCH_VERSION" ]]; then
  echo "  ✓ internal-dispatch-secret already populated"
else
  DISPATCH_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  echo -n "$DISPATCH_SECRET" | gcloud secrets versions add internal-dispatch-secret \
    --project="$PROJECT" \
    --data-file=- \
    --quiet
  echo "  ✓ internal-dispatch-secret set"
fi

# ── Instructions ─────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete."
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Any secret marked '!' above must be populated manually:"
echo "  echo -n 'VALUE' | gcloud secrets versions add SECRET_NAME --data-file=-"
echo ""
echo "  Once secrets are populated, run: GCP_PROJECT=$PROJECT ./scripts/deploy.sh"
