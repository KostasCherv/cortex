create table if not exists public.rag_chat_session_attachments (
    id uuid primary key,
    session_id uuid not null references public.rag_chat_sessions(id) on delete cascade,
    agent_id uuid not null,
    owner_id uuid not null,
    workspace_id text not null,
    resource_id uuid not null,
    filename text not null,
    mime_type text not null,
    byte_size bigint not null,
    storage_uri text not null,
    state text not null default 'uploaded'
        check (state in ('uploaded', 'processing', 'ready', 'failed')),
    error_details text null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_rag_chat_session_attachments_session_owner_agent_created
    on public.rag_chat_session_attachments(session_id, owner_id, agent_id, created_at asc);

create index if not exists idx_rag_chat_session_attachments_state
    on public.rag_chat_session_attachments(state);
