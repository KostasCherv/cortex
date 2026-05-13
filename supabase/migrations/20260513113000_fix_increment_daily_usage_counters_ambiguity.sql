create or replace function public.increment_daily_usage_counters(
    p_user_id uuid,
    p_usage_date date,
    p_research_queries integer,
    p_total_questions integer
)
returns table (
    user_id uuid,
    usage_date date,
    research_queries_count integer,
    total_questions_count integer,
    created_at timestamptz,
    updated_at timestamptz
)
language plpgsql
as $$
begin
    insert into public.daily_usage_counters as duc (
        user_id,
        usage_date,
        research_queries_count,
        total_questions_count
    )
    values (
        p_user_id,
        p_usage_date,
        greatest(0, p_research_queries),
        greatest(0, p_total_questions)
    )
    on conflict (user_id, usage_date)
    do update set
        research_queries_count = duc.research_queries_count + greatest(0, p_research_queries),
        total_questions_count = duc.total_questions_count + greatest(0, p_total_questions),
        updated_at = now();

    return query
    select
        q.user_id,
        q.usage_date,
        q.research_queries_count,
        q.total_questions_count,
        q.created_at,
        q.updated_at
    from public.daily_usage_counters as q
    where q.user_id = p_user_id
      and q.usage_date = p_usage_date;
end;
$$;
