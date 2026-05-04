alter table public.rag_chat_messages
    add column if not exists suggestions jsonb not null default '[]'::jsonb;
