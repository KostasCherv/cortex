-- Session attachments use resource_id as a graph-ingestion key only; they do not
-- create rows in rag_resources (see ingest_agent_chat_session_uploads).
alter table public.rag_chat_session_attachments
    drop constraint if exists rag_chat_session_attachments_resource_id_fkey;
