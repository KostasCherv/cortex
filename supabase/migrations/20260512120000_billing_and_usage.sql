create table if not exists public.user_subscriptions (
    user_id uuid primary key,
    plan text not null default 'free' check (plan in ('free', 'pro')),
    status text not null default 'inactive',
    stripe_customer_id text null,
    stripe_subscription_id text null,
    current_period_start timestamptz null,
    current_period_end timestamptz null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_user_subscriptions_stripe_customer
    on public.user_subscriptions(stripe_customer_id);

create index if not exists idx_user_subscriptions_stripe_subscription
    on public.user_subscriptions(stripe_subscription_id);

create table if not exists public.daily_usage_counters (
    user_id uuid not null,
    usage_date date not null,
    research_queries_count integer not null default 0 check (research_queries_count >= 0),
    total_questions_count integer not null default 0 check (total_questions_count >= 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (user_id, usage_date)
);

create index if not exists idx_daily_usage_user_date
    on public.daily_usage_counters(user_id, usage_date);

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
    insert into public.daily_usage_counters (
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
        research_queries_count = daily_usage_counters.research_queries_count + greatest(0, p_research_queries),
        total_questions_count = daily_usage_counters.total_questions_count + greatest(0, p_total_questions),
        updated_at = now();

    return query
    select
        duc.user_id,
        duc.usage_date,
        duc.research_queries_count,
        duc.total_questions_count,
        duc.created_at,
        duc.updated_at
    from public.daily_usage_counters duc
    where duc.user_id = p_user_id and duc.usage_date = p_usage_date;
end;
$$;
