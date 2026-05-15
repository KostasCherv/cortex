alter table public.rag_chat_sessions
    alter column agent_id drop not null;

alter table public.rag_chat_messages
    alter column agent_id drop not null;

alter table public.rag_chat_sessions
    add column if not exists chat_scope text not null default 'agent'
    check (chat_scope in ('agent', 'workspace'));

create index if not exists idx_rag_chat_sessions_owner_scope_created
    on public.rag_chat_sessions(owner_id, chat_scope, created_at desc);
