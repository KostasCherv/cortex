create table if not exists public.software_dev_plans (
    id uuid primary key,
    owner_id uuid not null,
    workspace_id text not null,
    prompt text not null,
    prompt_preview text not null,
    title text not null,
    summary text not null default '',
    suggested_filename text not null,
    markdown text not null,
    plan_json jsonb not null,
    planning_brief_json jsonb not null,
    repo_analysis_json jsonb not null,
    planning_options_json jsonb not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_software_dev_plans_owner_workspace_created
    on public.software_dev_plans(owner_id, workspace_id, created_at desc);
