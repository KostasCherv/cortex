alter table public.user_subscriptions
    add column if not exists cancel_at_period_end boolean null,
    add column if not exists cancel_at timestamptz null,
    add column if not exists canceled_at timestamptz null;
