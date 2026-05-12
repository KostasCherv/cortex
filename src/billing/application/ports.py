from __future__ import annotations

from datetime import date
from typing import Protocol

from src.billing.domain.models import DailyUsage, UserSubscription


class SubscriptionRepository(Protocol):
    async def get_user_subscription(self, user_id: str) -> UserSubscription | None: ...
    async def get_user_subscription_by_customer_id(self, customer_id: str) -> UserSubscription | None: ...
    async def get_user_subscription_by_subscription_id(self, subscription_id: str) -> UserSubscription | None: ...

    async def upsert_user_subscription(self, payload: dict) -> None: ...


class UsageRepository(Protocol):
    async def get_daily_usage(self, user_id: str, usage_date: date) -> DailyUsage | None: ...

    async def increment_daily_usage(
        self,
        *,
        user_id: str,
        usage_date: date,
        add_research_queries: int,
        add_total_questions: int,
    ) -> DailyUsage: ...


class StripeGateway(Protocol):
    async def create_checkout_session(self, *, user_id: str, email: str | None) -> str: ...

    async def create_portal_session(self, *, customer_id: str) -> str: ...

    def construct_webhook_event(self, payload: bytes, signature: str) -> dict: ...
