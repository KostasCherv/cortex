# Production deployment

Cortex deploys the backend to Google Cloud Run and the frontend to Vercel. Inngest calls the backend's serve endpoint for background functions, while Supabase provides managed data, authentication, and storage.

## Release path

The preferred backend release path is **GitHub Actions → Deploy production backend → Run workflow**.

The workflow:

1. Accepts only the `main` branch.
2. Runs the complete CI suite.
3. Waits for approval through the `Production` environment.
4. Applies pending Supabase migrations.
5. Authenticates to Google Cloud through GitHub OIDC.
6. Deploys the Cloud Run backend.
7. Runs post-deployment smoke checks.

Only one production deployment runs at a time.

## GitHub environment

The `Production` environment requires these variables:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
- `SUPABASE_PROJECT_REF`

Required secrets:

- `SUPABASE_ACCESS_TOKEN`
- `SUPABASE_DB_PASSWORD`

`SMOKE_TEST_TOKEN` is optional. When present, the workflow also checks an authenticated `GET /sessions` request. The token is never printed.

See [Production configuration](env-vars-production.md) for application variables and Secret Manager values.

## Manual backend deployment

The local path remains available for recovery and maintenance:

```bash
GCP_PROJECT=<project-id> ./scripts/setup_secrets.sh   # first deployment only
GCP_PROJECT=<project-id> ./scripts/deploy.sh
```

The deployment script:

1. Builds a container with Cloud Build and tags it with the current Git SHA.
2. Adds the `latest` tag.
3. Injects the immutable SHA tag into `cloudrun/service.yaml`.
4. Replaces the Cloud Run service.
5. Runs fail-fast smoke checks against liveness, readiness, authentication, and a one-event SSE probe.

Run smoke checks independently with:

```bash
python3 scripts/post_deploy_smoke.py https://<service-url>
```

## Health and readiness

- `GET /health` is a process-only liveness check and does not contact dependencies.
- `GET /ready` checks LLM configuration, Supabase, Neo4j, and optional Redis with per-dependency timeouts.
- A critical dependency failure returns HTTP 503.
- An optional Redis failure returns HTTP 200 with `status: degraded`.

Cloud Run uses `/ready` for startup/readiness and `/health` for liveness. The startup probe runs every ten seconds with a five-second timeout and permits 30 failures, giving a new revision up to five minutes to initialize.

Production normally sets `READINESS_REQUIRE_SUPABASE=true` and `READINESS_REQUIRE_NEO4J=true`. Both default to `false` for flexible local development.

For the metrics, alerts, and uptime checks built on these probes, see [Production monitoring](observability.md#production-monitoring).

### Neo4j Aura keep-alive

AuraDB Free pauses after 72 hours without query activity, which takes `/ready` (and therefore the whole service) down. Two things prevent this:

- The `/ready` Neo4j check runs a real `RETURN 1` query, not just a driver handshake, so each probe counts as Aura activity.
- A Cloud Scheduler job pings `/ready` every 12 hours, covering periods when the service is scaled to zero. Recreate it if needed:

```bash
gcloud scheduler jobs create http neo4j-keepalive \
  --project=<project> --location=us-central1 \
  --schedule="0 */12 * * *" \
  --uri="https://<service-url>/ready" \
  --http-method=GET --attempt-deadline=60s
```

## Rollback

The deploy script does not move traffic after a failed smoke check. To restore a known-good revision:

```bash
gcloud run revisions list \
  --service=cortex \
  --region=<region> \
  --project=<project>

gcloud run services update-traffic cortex \
  --region=<region> \
  --project=<project> \
  --to-revisions=<previous-revision>=100
```

## Frontend deployment

Vercel's Git integration deploys the production UI automatically. The backend workflow does not invoke Vercel or require Vercel credentials.

Manual environment synchronization and recovery deployment remain available:

```bash
./scripts/vercel-ui-env.sh
./scripts/deploy-ui.sh --prod
```

Set `CORS_ORIGINS` on the backend to a JSON array containing the production UI origin, for example `'["https://your-app.vercel.app"]'`.

## Background functions

After deploying the backend, configure the Inngest Dashboard serve URL:

```text
https://<cloud-run-url>/api/inngest
```

Registered functions:

- `rag-ingestion` — triggered by `rag/ingestion.requested`
- `research-run` — triggered by `research/run.requested`
- `outbox-dispatcher` — scheduled every two minutes

After syncing the functions, enable the production failure and missing-dispatcher email alerts described in [Production monitoring](observability.md#inngest-alert-activation).

## Alerting activation

Provision or refresh the Cloud Monitoring baseline after the backend service exists:

```bash
GCP_PROJECT=<project-id> ALERT_EMAIL=<operator-email> ./scripts/setup_alerting.sh --dry-run
GCP_PROJECT=<project-id> ALERT_EMAIL=<operator-email> ./scripts/setup_alerting.sh
```

The script configures the `/health` uptime check, 5xx burst policy, runtime/probe log alert, and email notification channel. Verify the email channel and send a test notification before treating the alarms as operational.

## Supabase migrations

The production workflow applies migrations automatically. For manual maintenance:

```bash
npx supabase link --project-ref <project-ref>
npx supabase db push
```

## Supply-chain controls

Every pull request and push to `main` builds the production container. CI scans the repository and image with Trivy, blocking HIGH or CRITICAL findings that have an available fix as well as serious configuration and secret findings.

CI also generates a CycloneDX container SBOM named `cortex-sbom-<commit-sha>` and retains it for 30 days. See [Testing and quality](testing-and-quality.md) for the complete release gates.
