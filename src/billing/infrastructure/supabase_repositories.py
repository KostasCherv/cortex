from __future__ import annotations

from datetime import date, datetime

from src.billing.application.ports import SubscriptionRepository, UsageRepository
from src.billing.domain.models import DailyUsage, Plan, UserSubscription
from src.db.supabase_store import SupabaseSessionStore


class SupabaseSubscriptionRepository(SubscriptionRepository):
    def __init__(self, store: SupabaseSessionStore) -> None:
        self._store = store

    async def get_user_subscription(self, user_id: str) -> UserSubscription | None:
        row = await self._store.get_user_subscription(user_id)
        return _row_to_subscription(row)

    async def get_user_subscription_by_customer_id(self, customer_id: str) -> UserSubscription | None:
        row = await self._store.get_user_subscription_by_customer_id(customer_id)
        return _row_to_subscription(row)

    async def get_user_subscription_by_subscription_id(self, subscription_id: str) -> UserSubscription | None:
        row = await self._store.get_user_subscription_by_subscription_id(subscription_id)
        return _row_to_subscription(row)

    async def upsert_user_subscription(self, payload: dict) -> None:
        await self._store.upsert_user_subscription(payload)


def _row_to_subscription(row: dict | None) -> UserSubscription | None:
        if not row:
            return None
        plan_raw = str(row.get("plan") or "free")
        plan = Plan.PRO if plan_raw == Plan.PRO.value else Plan.FREE
        return UserSubscription(
            user_id=row["user_id"],
            plan=plan,
            status=row.get("status") or "inactive",
            stripe_customer_id=row.get("stripe_customer_id"),
            stripe_subscription_id=row.get("stripe_subscription_id"),
            current_period_start=_parse_datetime(row.get("current_period_start")),
            current_period_end=_parse_datetime(row.get("current_period_end")),
            cancel_at_period_end=row.get("cancel_at_period_end"),
            cancel_at=_parse_datetime(row.get("cancel_at")),
            canceled_at=_parse_datetime(row.get("canceled_at")),
        )


class SupabaseUsageRepository(UsageRepository):
    def __init__(self, store: SupabaseSessionStore) -> None:
        self._store = store

    async def get_daily_usage(self, user_id: str, usage_date: date) -> DailyUsage | None:
        row = await self._store.get_daily_usage_counter(user_id=user_id, usage_date=usage_date)
        if not row:
            return None
        return DailyUsage(
            usage_date=usage_date,
            research_queries_count=int(row.get("research_queries_count") or 0),
            total_questions_count=int(row.get("total_questions_count") or 0),
        )

    async def increment_daily_usage(
        self,
        *,
        user_id: str,
        usage_date: date,
        add_research_queries: int,
        add_total_questions: int,
    ) -> DailyUsage:
        row = await self._store.increment_daily_usage_counter(
            user_id=user_id,
            usage_date=usage_date,
            add_research_queries=add_research_queries,
            add_total_questions=add_total_questions,
        )
        return DailyUsage(
            usage_date=usage_date,
            research_queries_count=int(row.get("research_queries_count") or 0),
            total_questions_count=int(row.get("total_questions_count") or 0),
        )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
