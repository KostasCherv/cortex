alter table public.rag_chat_sessions
add column if not exists web_search_enabled boolean not null default false;

create index if not exists idx_rag_chat_sessions_owner_agent_web_search
    on public.rag_chat_sessions(owner_id, agent_id, web_search_enabled);
