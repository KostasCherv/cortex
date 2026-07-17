#!/usr/bin/env bash
# Create or update the low-cost production alerting baseline.
#
# Usage:
#   GCP_PROJECT=my-project ALERT_EMAIL=ops@example.com ./scripts/setup_alerting.sh
#   GCP_PROJECT=my-project ALERT_EMAIL=ops@example.com ./scripts/setup_alerting.sh --dry-run

set -euo pipefail

PROJECT="${GCP_PROJECT:-}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${GCP_SERVICE:-cortex}"
EMAIL="${ALERT_EMAIL:-}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif [[ -n "${1:-}" ]]; then
  echo "ERROR: unknown argument: $1" >&2
  exit 2
fi

if [[ -z "$PROJECT" || -z "$EMAIL" ]]; then
  echo "ERROR: GCP_PROJECT and ALERT_EMAIL are required." >&2
  echo "Usage: GCP_PROJECT=my-project ALERT_EMAIL=ops@example.com $0 [--dry-run]" >&2
  exit 2
fi
if [[ "$EMAIL" != *@*.* ]]; then
  echo "ERROR: ALERT_EMAIL does not look like an email address." >&2
  exit 2
fi

for command in gcloud curl python3; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $command" >&2
    exit 1
  }
done

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cortex-alerting.XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Project : $PROJECT"
echo "Region  : $REGION"
echo "Service : $SERVICE"
echo "Email   : $EMAIL"
echo "Dry run : $DRY_RUN"

if $DRY_RUN; then
  echo "DRY RUN: would enable monitoring.googleapis.com and logging.googleapis.com"
else
  gcloud services enable monitoring.googleapis.com logging.googleapis.com \
    --project="$PROJECT" --quiet
fi

SERVICE_URL=$(gcloud run services describe "$SERVICE" \
  --project="$PROJECT" \
  --region="$REGION" \
  --format="value(status.url)")
if [[ -z "$SERVICE_URL" ]]; then
  echo "ERROR: Cloud Run service URL was empty for $SERVICE." >&2
  exit 1
fi
HOST=${SERVICE_URL#https://}
HOST=${HOST#http://}
HOST=${HOST%%/*}

# Notification channels are not exposed by the GA gcloud command group, so use
# the Monitoring REST API while still relying on the active gcloud identity.
CHANNEL_NAME="projects/$PROJECT/notificationChannels/DRY_RUN"
CHANNEL_STATUS="VERIFIED"
if $DRY_RUN; then
  echo "DRY RUN: would upsert email channel cortex-ops-email for $EMAIL"
else
  TOKEN=$(gcloud auth print-access-token)
  CHANNELS_FILE="$TMP_DIR/channels.json"
  curl -fsS \
    -H "Authorization: Bearer $TOKEN" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT/notificationChannels" \
    > "$CHANNELS_FILE"

  CHANNEL_INFO=$(python3 - "$CHANNELS_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for channel in payload.get("notificationChannels", []):
    if channel.get("displayName") == "cortex-ops-email":
        labels = channel.get("labels", {})
        print(channel.get("name", ""), labels.get("email_address", ""), channel.get("verificationStatus", ""), sep="\t")
        break
PY
  )

  CHANNEL_NAME=$(printf '%s' "$CHANNEL_INFO" | cut -f1)
  EXISTING_EMAIL=$(printf '%s' "$CHANNEL_INFO" | cut -f2)
  if [[ -z "$CHANNEL_NAME" ]]; then
    python3 - "$TMP_DIR/channel.json" "$EMAIL" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({
        "type": "email",
        "displayName": "cortex-ops-email",
        "labels": {"email_address": sys.argv[2]},
        "enabled": True,
    }, handle)
PY
    curl -fsS -X POST \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      --data-binary "@$TMP_DIR/channel.json" \
      "https://monitoring.googleapis.com/v3/projects/$PROJECT/notificationChannels" \
      > "$TMP_DIR/channel-response.json"
  elif [[ "$EXISTING_EMAIL" != "$EMAIL" ]]; then
    python3 - "$TMP_DIR/channel.json" "$EMAIL" <<'PY'
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({
        "displayName": "cortex-ops-email",
        "labels": {"email_address": sys.argv[2]},
        "enabled": True,
    }, handle)
PY
    curl -fsS -X PATCH \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      --data-binary "@$TMP_DIR/channel.json" \
      "https://monitoring.googleapis.com/v3/$CHANNEL_NAME?updateMask=display_name,labels,enabled" \
      > "$TMP_DIR/channel-response.json"
  else
    cp "$CHANNELS_FILE" "$TMP_DIR/channel-response.json"
  fi

  if [[ -f "$TMP_DIR/channel-response.json" && "$CHANNEL_NAME" == "" ]]; then
    CHANNEL_NAME=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("name", ""))' "$TMP_DIR/channel-response.json")
  fi
  if [[ -z "$CHANNEL_NAME" ]]; then
    echo "ERROR: failed to create or find cortex-ops-email notification channel." >&2
    exit 1
  fi

  # Re-read so verification status reflects the server's current state.
  curl -fsS \
    -H "Authorization: Bearer $TOKEN" \
    "https://monitoring.googleapis.com/v3/$CHANNEL_NAME" \
    > "$TMP_DIR/channel-current.json"
  CHANNEL_STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verificationStatus", ""))' "$TMP_DIR/channel-current.json")
fi

UPTIME_DISPLAY_NAME="cortex-backend-uptime"
UPTIME_NAME=""
if ! $DRY_RUN; then
  gcloud monitoring uptime list-configs --project="$PROJECT" --format=json > "$TMP_DIR/uptime-list.json"
  UPTIME_NAME=$(python3 - "$TMP_DIR/uptime-list.json" "$UPTIME_DISPLAY_NAME" <<'PY'
import json
import sys

for check in json.load(open(sys.argv[1], encoding="utf-8")):
    if check.get("displayName") == sys.argv[2]:
        print(check.get("name", ""))
        break
PY
  )
fi

if $DRY_RUN; then
  echo "DRY RUN: would upsert $UPTIME_DISPLAY_NAME for https://$HOST/health"
  UPTIME_ID="DRY_RUN"
elif [[ -z "$UPTIME_NAME" ]]; then
  UPTIME_NAME=$(gcloud monitoring uptime create "$UPTIME_DISPLAY_NAME" \
    --project="$PROJECT" \
    --resource-type=uptime-url \
    --resource-labels="host=$HOST,project_id=$PROJECT" \
    --protocol=https \
    --port=443 \
    --path=/health \
    --request-method=get \
    --validate-ssl=true \
    --status-classes=2xx \
    --matcher-content='"ok"' \
    --matcher-type=matches-json-path \
    --json-path='$.status' \
    --json-path-matcher-type=exact-match \
    --period=1 \
    --timeout=10 \
    --regions=usa-iowa,europe,asia-pacific \
    --user-labels="service=$SERVICE,managed_by=cortex" \
    --format="value(name)")
  UPTIME_ID=${UPTIME_NAME##*/}
else
  UPTIME_ID=${UPTIME_NAME##*/}
  gcloud monitoring uptime update "$UPTIME_ID" \
    --project="$PROJECT" \
    --display-name="$UPTIME_DISPLAY_NAME" \
    --path=/health \
    --port=443 \
    --request-method=get \
    --validate-ssl=true \
    --set-status-classes=2xx \
    --matcher-content='"ok"' \
    --matcher-type=matches-json-path \
    --json-path='$.status' \
    --json-path-matcher-type=exact-match \
    --period=1 \
    --timeout=10 \
    --set-regions=usa-iowa,europe,asia-pacific \
    --update-user-labels="service=$SERVICE,managed_by=cortex" \
    --quiet
fi

python3 - "$TMP_DIR" "$PROJECT" "$SERVICE" "$CHANNEL_NAME" "$UPTIME_ID" <<'PY'
import json
import os
import sys

output_dir, project, service, channel, uptime_id = sys.argv[1:]

common = {
    "combiner": "OR",
    "enabled": True,
    "notificationChannels": [channel],
    "userLabels": {"service": service, "managed_by": "cortex"},
}

policies = {
    "cortex-backend-uptime.json": {
        **common,
        "displayName": "cortex-backend-uptime",
        "documentation": {
            "mimeType": "text/markdown",
            "content": "The Cortex /health endpoint failed from at least two regions. Check Cloud Run revisions and system logs before rollback.",
        },
        "conditions": [{
            "displayName": "Two regions cannot reach /health",
            "conditionThreshold": {
                "filter": f'resource.type = "uptime_url" AND metric.type = "monitoring.googleapis.com/uptime_check/check_passed" AND metric.label.check_id = "{uptime_id}"',
                "comparison": "COMPARISON_LT",
                "thresholdValue": 1,
                "duration": "60s",
                "aggregations": [{"alignmentPeriod": "60s", "perSeriesAligner": "ALIGN_NEXT_OLDER"}],
                "trigger": {"count": 2},
            },
        }],
        "alertStrategy": {"autoClose": "1800s"},
    },
    "cortex-5xx-burst.json": {
        **common,
        "displayName": "cortex-5xx-burst",
        "documentation": {
            "mimeType": "text/markdown",
            "content": "Cortex returned at least three 5xx responses in five minutes. Inspect Cloud Logging and Sentry for the matching production revision.",
        },
        "conditions": [{
            "displayName": "At least three 5xx responses in five minutes",
            "conditionThreshold": {
                "filter": f'resource.type = "cloud_run_revision" AND resource.label.service_name = "{service}" AND metric.type = "run.googleapis.com/request_count" AND metric.label.response_code_class = "5xx"',
                "comparison": "COMPARISON_GT",
                "thresholdValue": 2,
                "duration": "0s",
                "aggregations": [{
                    "alignmentPeriod": "300s",
                    "perSeriesAligner": "ALIGN_DELTA",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": ["resource.label.service_name"],
                }],
                "trigger": {"count": 1},
            },
        }],
        "alertStrategy": {"autoClose": "1800s"},
    },
    "cortex-runtime-failure.json": {
        **common,
        "displayName": "cortex-runtime-failure",
        "documentation": {
            "mimeType": "text/markdown",
            "content": "Cloud Run reported a container termination or failed startup/liveness/readiness probe for Cortex. Inspect the revision's system logs and /ready dependency statuses.",
        },
        "conditions": [{
            "displayName": "Container or probe failure",
            "conditionMatchedLog": {
                "filter": (
                    f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}" '
                    'AND severity>=ERROR AND ('
                    'log_id("run.googleapis.com/varlog/system") OR '
                    'textPayload=~"(?i)(startup|readiness|liveness) probe.*(fail|error|timeout)" OR '
                    'textPayload=~"(?i)container.*(exit|terminat|crash)" OR '
                    'jsonPayload.message=~"(?i)(startup|readiness|liveness) probe.*(fail|error|timeout)"'
                    ')'
                ),
            },
        }],
        "alertStrategy": {
            "notificationRateLimit": {"period": "900s"},
            "autoClose": "1800s",
        },
    },
}

for filename, policy in policies.items():
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2)
        handle.write("\n")
PY

upsert_policy() {
  local display_name="$1"
  local policy_file="$2"
  local existing=""
  if ! $DRY_RUN; then
    gcloud monitoring policies list --project="$PROJECT" --format=json \
      > "$TMP_DIR/policies.json"
    existing=$(python3 - "$TMP_DIR/policies.json" "$display_name" <<'PY'
import json
import sys

for policy in json.load(open(sys.argv[1], encoding="utf-8")):
    if policy.get("displayName") == sys.argv[2]:
        print(policy.get("name", ""))
        break
PY
    )
  fi

  if $DRY_RUN; then
    echo "DRY RUN: would upsert alert policy $display_name"
  elif [[ -n "$existing" ]]; then
    gcloud monitoring policies update "$existing" \
      --project="$PROJECT" \
      --policy-from-file="$policy_file" \
      --quiet >/dev/null
    echo "Updated policy: $display_name"
  else
    gcloud monitoring policies create \
      --project="$PROJECT" \
      --policy-from-file="$policy_file" \
      --quiet >/dev/null
    echo "Created policy: $display_name"
  fi
}

upsert_policy "cortex-backend-uptime" "$TMP_DIR/cortex-backend-uptime.json"
upsert_policy "cortex-5xx-burst" "$TMP_DIR/cortex-5xx-burst.json"
upsert_policy "cortex-runtime-failure" "$TMP_DIR/cortex-runtime-failure.json"

if $DRY_RUN; then
  echo "Dry run complete; no cloud resources were changed."
  exit 0
fi

if [[ "$CHANNEL_STATUS" == "UNVERIFIED" ]]; then
  echo "ERROR: alert resources were created, but cortex-ops-email is unverified." >&2
  echo "Verify $EMAIL in Cloud Monitoring → Alerting → Notification channels, then rerun this script." >&2
  exit 2
fi

echo "Alerting baseline is configured."
echo "Test the channel in Cloud Monitoring and confirm the /health uptime check is passing."
