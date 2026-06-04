create table if not exists public.user_memory (
    owner_id uuid not null,
    workspace_id text not null,
    content text not null,
    updated_at timestamptz not null default now(),
    last_refreshed_at timestamptz null,
    primary key (owner_id, workspace_id)
);

create table if not exists public.user_memory_refresh_events (
    id uuid primary key,
    owner_id uuid not null,
    workspace_id text not null,
    event_key text not null unique,
    source_mode text not null,
    source_session_id text not null,
    source_user_message_id text null,
    source_assistant_message_id text null,
    processed_at timestamptz not null default now(),
    created_at timestamptz not null default now()
);

create index if not exists idx_user_memory_refresh_events_owner_workspace
    on public.user_memory_refresh_events(owner_id, workspace_id, created_at desc);
