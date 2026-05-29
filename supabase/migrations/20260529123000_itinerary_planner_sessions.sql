create table if not exists public.itinerary_sessions (
    id uuid primary key,
    owner_id uuid not null,
    workspace_id text not null,
    title text not null default 'New itinerary',
    status text not null default 'collecting_requirements',
    requirements_json jsonb not null default '{}'::jsonb,
    current_version_id uuid null,
    prompt_preview text not null default '',
    last_message_preview text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.itinerary_messages (
    id uuid primary key,
    session_id uuid not null references public.itinerary_sessions(id) on delete cascade,
    owner_id uuid not null,
    role text not null,
    content text not null,
    metadata_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.itinerary_versions (
    id uuid primary key,
    session_id uuid not null references public.itinerary_sessions(id) on delete cascade,
    owner_id uuid not null,
    version_number integer not null,
    revision_summary text not null,
    markdown text not null,
    itinerary_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_itinerary_sessions_owner_workspace_updated
    on public.itinerary_sessions(owner_id, workspace_id, updated_at desc);

create index if not exists idx_itinerary_messages_owner_session_created
    on public.itinerary_messages(owner_id, session_id, created_at asc);

create index if not exists idx_itinerary_versions_owner_session_created
    on public.itinerary_versions(owner_id, session_id, created_at desc);
