# Production Environment Variables

## Backend — Google Cloud Run

All secrets should be stored in **Google Secret Manager** and referenced via `valueFrom.secretKeyRef` in `cloudrun/service.yaml`. Non-sensitive config can be set as literal `value` entries.

### Required

| Env Var | Secret Manager | Notes |
|---|---|---|
| `OPENAI_API_KEY` | yes | Required when `LLM_PROVIDER=openai` |
| `OPENROUTER_API_KEY` | yes | Required when `LLM_PROVIDER=openrouter` |
| `LLM_PROVIDER` | no | `openai` or `openrouter` |
| `OPENAI_MODEL` | no | e.g. `gpt-4o-mini` |
| `EMBEDDING_PROVIDER` | no | `openai` |
| `EMBEDDING_MODEL` | no | `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | no | `1536` |
| `TAVILY_API_KEY` | yes | |
| `ALPHA_VANTAGE_API_KEY` | yes | Required when `ASSET_PRICE_PROVIDER=alphavantage_mcp` |
| `ASSET_PRICE_PROVIDER` | no | `alphavantage_mcp` |
| `ALPHA_VANTAGE_MCP_URL` | no | Optional full remote MCP URL override |
| `ALPHA_VANTAGE_MCP_TOOL_REFRESH_SECONDS` | no | Tool-catalog refresh cadence in seconds (default `3600`) |
| `NEO4J_URI` | yes | Use `neo4j+s://` for AuraDB (TLS) |
| `NEO4J_USERNAME` | yes | |
| `NEO4J_PASSWORD` | yes | |
| `NEO4J_DATABASE` | no | `neo4j` |
| `SUPABASE_URL` | no | `https://<project>.supabase.co` |
| `SUPABASE_SECRET_KEY` | yes | `sb_secret_...` from Dashboard → Settings → API Keys (`SUPABASE_SERVICE_ROLE_KEY` still accepted) |
| `SUPABASE_JWKS_URL` | no | `https://<project>.supabase.co/auth/v1/.well-known/jwks.json` |
| `SUPABASE_JWT_SECRET` | yes | |
| `READINESS_REQUIRE_SUPABASE` | no | Set to `true` in production; `/ready` returns `503` if Supabase is missing or unavailable |
| `READINESS_REQUIRE_NEO4J` | no | Set to `true` in production; `/ready` returns `503` if Neo4j is missing or unavailable |
| `READINESS_TIMEOUT_SECONDS` | no | Per-dependency probe timeout; defaults to `2.0` seconds |
| `INNGEST_EVENT_KEY` | yes | From Inngest dashboard → Keys |
| `INNGEST_SIGNING_KEY` | yes | From Inngest dashboard → Keys. **Must not be empty in prod.** |
| `REDIS_URL` | yes | Upstash `rediss://` URL |
| `STRIPE_SECRET_KEY` | yes | |
| `STRIPE_WEBHOOK_SECRET` | yes | Update Stripe webhook URL to `https://<cloud-run-url>/api/billing/webhook` after first deploy |
| `STRIPE_PRO_PRICE_ID` | no | |
| `STRIPE_SUCCESS_URL` | no | `https://<frontend>/billing/success` |
| `STRIPE_CANCEL_URL` | no | `https://<frontend>/billing/cancel` |
| `STRIPE_PORTAL_RETURN_URL` | no | `https://<frontend>` |
| `CORS_ORIGINS` | no | `https://<frontend>` — no trailing slash, no wildcard |
| `INTERNAL_DISPATCH_SECRET` | yes | Random 32-byte hex: `python -c "import secrets; print(secrets.token_hex(32))"` |

### Optional (recommended)

| Env Var | Secret Manager | Notes |
|---|---|---|
| `COHERE_API_KEY` | yes | Enables cross-encoder reranking |
| `LANGFUSE_PUBLIC_KEY` | yes | |
| `LANGFUSE_SECRET_KEY` | yes | |
| `LANGFUSE_ENV` | no | `prod` |
| `LANGFUSE_RELEASE` | no | Git SHA or semver, e.g. `v1.2.3` |
| `SENTRY_DSN` | yes | Error tracking; unset disables it entirely |
| `RATE_LIMIT_DEFAULT` | no | Per-IP request limit, e.g. `60/minute` (default) |

### Variables that must NOT be set in production

| Env Var | Why |
|---|---|
| `INNGEST_DEV` | Setting to `1` disables Inngest signature verification — allows unauthenticated event injection |

---

## Deploy commands

```bash
# 1. Build and push image
gcloud builds submit --tag gcr.io/cortex-496709/cortex:latest .

# 2. Deploy (replace REGION with your preferred region, e.g. us-central1)
gcloud run services replace cloudrun/service.yaml --region=REGION

# 3. Allow public access
gcloud run services add-iam-policy-binding cortex \
  --region=REGION \
  --member="allUsers" \
  --role="roles/run.invoker"

# 4. Get the service URL
gcloud run services describe cortex --region=REGION --format="value(status.url)"
```

### Create all secrets (run once)

```bash
PROJECT=cortex-496709
REGION=your-region

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

for secret in "${secrets[@]}"; do
  echo "Creating secret: $secret"
  echo -n "PLACEHOLDER" | gcloud secrets create "$secret" \
    --project="$PROJECT" \
    --data-file=- \
    --replication-policy=automatic 2>/dev/null || echo "  (already exists, skipping)"
done

# Then update each secret with its real value:
# echo -n "sk-..." | gcloud secrets versions add openai-api-key --data-file=-
```

---

## Inngest post-deploy wiring

After the Cloud Run service is deployed:

1. Log into [app.inngest.com](https://app.inngest.com)
2. Go to your `cortex` app
3. Set **Serve URL** to `https://<cloud-run-url>/api/inngest`
4. Confirm **Event Key** and **Signing Key** match what was set in Cloud Run secrets
5. Click **Sync** — all three functions should appear: `rag-ingestion`, `research-run`, `outbox-dispatcher`
6. Verify `outbox-dispatcher` shows a cron trigger of `* * * * *`

---

## Frontend — Vercel

### Project settings (Vercel dashboard)

| Setting | Value |
|---|---|
| Framework Preset | Vite |
| Root Directory | `ui` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Node.js Version | 20.x |

### Environment variables (Production)

| Variable | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://<cloud-run-service-url>` (no trailing slash) |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Supabase publishable key (`sb_publishable_...`, safe to expose) |

> **Note:** Only variables prefixed with `VITE_` are injected into the browser bundle by Vite.

### Smoke test checklist

After deploying both backend and frontend:

- [ ] `curl https://<cloud-run-url>/health` returns `{"status":"ok",...}`
- [ ] Frontend loads at Vercel URL
- [ ] Sign in via Supabase auth works
- [ ] Create a research session and submit a query — SSE stream completes
- [ ] Upload a RAG resource — status moves to `ready` within ~1 minute (Inngest cron)
- [ ] Stripe test checkout redirects to `https://<frontend>/billing/success`
- [ ] `curl -X POST https://<cloud-run-url>/internal/dispatch-outbox -H "Authorization: Bearer wrongkey"` returns 401
