alter table if exists public.session_runs
    add column if not exists latest_node text null;

alter table if exists public.session_runs
    add column if not exists latest_event_at timestamptz null;

alter table if exists public.session_runs
    add column if not exists partial_report text not null default '';
