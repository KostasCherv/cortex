alter table if exists public.session_runs
    add column if not exists langfuse_trace_id text null;

alter table if exists public.session_runs
    add column if not exists langfuse_observation_id text null;

alter table if exists public.session_runs
    add column if not exists feedback_submitted_at timestamptz null;

alter table if exists public.session_runs
    add column if not exists feedback_helpful boolean null;
