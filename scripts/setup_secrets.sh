#!/usr/bin/env bash
# First-time setup: creates the required secrets in Google Secret Manager.
# Run once before the first deploy.
#
# Usage: GCP_PROJECT=my-project ./scripts/setup_secrets.sh
#    or: ./scripts/setup_secrets.sh my-project
#
# If .env.prod exists at the repo root, secret values are populated from it
# automatically. Otherwise populate each secret manually:
#   echo -n "sk-..." | gcloud secrets versions add openai-api-key --data-file=-
#
# NOTE: the lean set below is 7 secrets. Secret Manager's free tier covers
# 6 active versions; the 7th costs ~$0.06/month. When rotating a value,
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
# Lean set: only real credentials. Non-sensitive config (Neo4j URI/username,
# Supabase URL) lives as plain env vars in cloudrun/service.yaml.
# Optional services (Redis, Stripe, Cohere, LangSmith, LangFuse) are disabled
# in this deploy; add their secrets back alongside service.yaml if enabled.

secrets=(
  openai-api-key
  tavily-api-key
  neo4j-password
  supabase-secret-key
  inngest-event-key
  inngest-signing-key
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

  populate openai-api-key OPENAI_API_KEY
  populate tavily-api-key TAVILY_API_KEY
  populate neo4j-password NEO4J_PASSWORD
  populate supabase-secret-key SUPABASE_SECRET_KEY
  populate inngest-event-key INNGEST_EVENT_KEY
  populate inngest-signing-key INNGEST_SIGNING_KEY
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
echo "  Setup complete."
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Any secret marked '!' above must be populated manually:"
echo "  echo -n 'VALUE' | gcloud secrets versions add SECRET_NAME --data-file=-"
echo ""
echo "  Once secrets are populated, run: GCP_PROJECT=$PROJECT ./scripts/deploy.sh"
