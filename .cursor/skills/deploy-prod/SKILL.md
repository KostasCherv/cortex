---
name: deploy-prod
description: >-
  Deploy cortex to production (Cloud Run backend + Vercel UI). Use when the user
  asks to deploy, ship, release, or push to prod, or invokes /deploy-prod.
trigger: /deploy-prod
---

# Deploy cortex to production

Full-stack production deploy for this repo. **Always read README.md § "Production deployment"** for canonical details; this skill orchestrates the existing scripts.

## Safety rules

1. **Confirm before deploying** — production deploys cost money and affect users. Ask once unless the user already said "deploy to prod" explicitly.
2. **Never print secrets** — `.env.prod` holds credentials. Do not `cat` it, log values, or commit it.
3. **Deploy current HEAD** — `scripts/deploy.sh` tags the image with `git rev-parse --short HEAD`. Warn if the working tree is dirty or unpushed.
4. **Do not skip hooks or force-push** unless the user explicitly requests it.

## Architecture

| Layer | Platform | Script |
|-------|----------|--------|
| Backend API | Google Cloud Run | `scripts/deploy.sh` |
| Frontend UI | Vercel | `scripts/deploy-ui.sh --prod` |
| Background jobs | Inngest (sync after backend) | manual dashboard step |
| Database | Supabase | `npx supabase db push` (when migrations changed) |

Secrets live in **Google Secret Manager** (backend) and **Vercel env** (UI `VITE_*` vars). Non-sensitive config is in `cloudrun/service.yaml`.

## Resolve GCP project

```bash
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
```

If empty, ask the user for their GCP project ID. Do not guess.

## Pre-flight checklist

Run these before deploying. Stop and report if any hard prerequisite fails.

```bash
# Tools
command -v gcloud >/dev/null && gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1
command -v git >/dev/null
npx --yes vercel@latest whoami 2>/dev/null

# Repo state
git status --short
git rev-parse --short HEAD

# Config files exist
test -f .env.prod && echo ".env.prod present" || echo "WARN: .env.prod missing"
test -f cloudrun/service.yaml
```

**First-time backend setup** (secrets not yet in GCP):

```bash
GCP_PROJECT="$GCP_PROJECT" ./scripts/setup_secrets.sh
```

Only run `setup_secrets.sh` when secrets are missing or the user asks for first-time setup.

## Deploy workflow

Copy and track progress:

```
Deploy progress:
- [ ] Pre-flight passed
- [ ] Backend deployed (Cloud Run)
- [ ] Backend health check OK
- [ ] UI env synced (if VITE_* changed)
- [ ] UI deployed (Vercel prod)
- [ ] Post-deploy reminders sent
```

### Step 1 — Backend (Cloud Run)

```bash
GCP_PROJECT="$GCP_PROJECT" ./scripts/deploy.sh
# Force full Docker rebuild (slow): add --no-cache
```

The script builds via Cloud Build, tags with git SHA, and runs `gcloud run services replace` using `cloudrun/service.yaml`.

Capture the service URL from script output, or:

```bash
gcloud run services describe cortex \
  --region="${GCP_REGION:-us-central1}" \
  --project="$GCP_PROJECT" \
  --format="value(status.url)"
```

### Step 2 — Health check

```bash
curl -sf "$SERVICE_URL/health" && echo " OK"
```

If health fails, pull Cloud Run logs before continuing:

```bash
gcloud run services logs read cortex \
  --region="${GCP_REGION:-us-central1}" \
  --project="$GCP_PROJECT" \
  --limit=50
```

### Step 3 — UI env sync (when needed)

Run when `VITE_*` values in `.env.prod` changed, or first UI deploy:

```bash
./scripts/vercel-ui-env.sh
```

Requires `ui/.vercel/project.json`. If missing:

```bash
./scripts/deploy-ui.sh --link-only
./scripts/vercel-ui-env.sh
```

### Step 4 — Frontend (Vercel production)

```bash
./scripts/deploy-ui.sh --prod
```

### Step 5 — Post-deploy (tell the user)

After a successful backend deploy, remind the user to verify:

1. **Inngest** — Apps → Sync → Serve URL: `https://<cloud-run-url>/api/inngest`
2. **CORS** — `CORS_ORIGINS` in `cloudrun/service.yaml` must include the Vercel prod URL
3. **Stripe webhook** (if billing enabled) — `https://<cloud-run-url>/api/billing/webhook`
4. **Supabase migrations** — if schema changed: `npx supabase link --project-ref <ref>` then `npx supabase db push`

## Partial deploys

| User wants | Run |
|------------|-----|
| Backend only | Steps 1–2 |
| UI only | Steps 3–4 |
| Full stack | All steps |
| First-time setup | `setup_secrets.sh` then full stack |

## Failure recovery

| Symptom | Action |
|---------|--------|
| `GCP_PROJECT not set` | Export `GCP_PROJECT` or set `gcloud config set project` |
| `gcloud CLI not found` | Install Google Cloud SDK |
| Cloud Build fails | Read build log; try `--no-cache` if stale layer |
| `Not logged in to Vercel` | `npx vercel login` |
| Health 5xx | Check Cloud Run logs; verify Secret Manager secrets populated |
| UI build fails | Fix locally with `cd ui && npm run build` first |

For Vercel-specific troubleshooting, also use the `deployment-expert` agent.

## Report template

When done, summarize:

```markdown
## Production deploy complete

- **Commit**: `<sha>`
- **Backend**: `<cloud-run-url>` (health: OK/FAIL)
- **Frontend**: `<vercel-url>` (if deployed)
- **Manual follow-ups**: Inngest sync, CORS, migrations (list any that apply)
```
