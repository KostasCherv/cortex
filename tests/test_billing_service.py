from __future__ import annotations

from datetime import date

import pytest

from src.billing.application.service import BillingService, UsageIncrement
from src.billing.domain.models import DailyUsage, Plan, UserSubscription


class _FakeSubscriptions:
    def __init__(self) -> None:
        self.by_user: dict[str, UserSubscription] = {}
        self.by_customer: dict[str, UserSubscription] = {}
        self.by_subscription: dict[str, UserSubscription] = {}
        self.last_upsert: dict | None = None

    async def get_user_subscription(self, user_id: str) -> UserSubscription | None:
        return self.by_user.get(user_id)

    async def get_user_subscription_by_customer_id(self, customer_id: str) -> UserSubscription | None:
        return self.by_customer.get(customer_id)

    async def get_user_subscription_by_subscription_id(self, subscription_id: str) -> UserSubscription | None:
        return self.by_subscription.get(subscription_id)

    async def upsert_user_subscription(self, payload: dict) -> None:
        self.last_upsert = payload


class _FakeUsage:
    async def get_daily_usage(self, user_id: str, usage_date: date) -> DailyUsage | None:
        return DailyUsage(usage_date=usage_date, research_queries_count=0, total_questions_count=0)

    async def increment_daily_usage(
        self,
        *,
        user_id: str,
        usage_date: date,
        add_research_queries: int,
        add_total_questions: int,
    ) -> DailyUsage:
        return DailyUsage(
            usage_date=usage_date,
            research_queries_count=add_research_queries,
            total_questions_count=add_total_questions,
        )


class _FakeStripe:
    def __init__(self, event: dict | None = None) -> None:
        self.event = event or {}
        self.subscriptions: dict[str, dict] = {}

    async def create_checkout_session(self, *, user_id: str, email: str | None) -> str:
        return "https://checkout"

    async def create_portal_session(self, *, customer_id: str) -> str:
        return "https://portal"

    async def get_subscription(self, subscription_id: str) -> dict | None:
        return self.subscriptions.get(subscription_id)

    def construct_webhook_event(self, payload: bytes, signature: str) -> dict:
        return self.event


@pytest.mark.asyncio
async def test_checkout_completed_uses_client_reference_id_when_metadata_missing():
    subs = _FakeSubscriptions()
    stripe = _FakeStripe(
        {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": "user-1",
                    "customer": "cus_123",
                    "subscription": "sub_123",
                    "metadata": {},
                }
            },
        }
    )
    service = BillingService(subscriptions=subs, usage=_FakeUsage(), stripe=stripe)

    await service.handle_webhook(b"{}", "sig")

    assert subs.last_upsert is not None
    assert subs.last_upsert["user_id"] == "user-1"
    assert subs.last_upsert["plan"] == "pro"


@pytest.mark.asyncio
async def test_checkout_completed_enriches_period_fields_from_subscription_lookup():
    subs = _FakeSubscriptions()
    stripe = _FakeStripe(
        {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": "user-42",
                    "customer": "cus_checkout",
                    "subscription": "sub_checkout",
                    "metadata": {},
                }
            },
        }
    )
    stripe.subscriptions["sub_checkout"] = {
        "id": "sub_checkout",
        "customer": "cus_real",
        "status": "trialing",
        "current_period_start": 1_700_000_000,
        "current_period_end": 1_700_100_000,
    }
    service = BillingService(subscriptions=subs, usage=_FakeUsage(), stripe=stripe)

    await service.handle_webhook(b"{}", "sig")

    assert subs.last_upsert is not None
    assert subs.last_upsert["user_id"] == "user-42"
    assert subs.last_upsert["status"] == "trialing"
    assert subs.last_upsert["stripe_customer_id"] == "cus_real"
    assert subs.last_upsert["stripe_subscription_id"] == "sub_checkout"
    assert subs.last_upsert["current_period_start"] is not None
    assert subs.last_upsert["current_period_end"] is not None


@pytest.mark.asyncio
async def test_subscription_event_resolves_user_by_subscription_id_when_metadata_missing():
    subs = _FakeSubscriptions()
    subs.by_subscription["sub_abc"] = UserSubscription(
        user_id="user-9",
        plan=Plan.PRO,
        status="active",
        stripe_subscription_id="sub_abc",
        stripe_customer_id="cus_abc",
    )

    stripe = _FakeStripe(
        {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_abc",
                    "customer": "cus_abc",
                    "status": "canceled",
                    "metadata": {},
                    "current_period_start": 1_700_000_000,
                    "current_period_end": 1_700_100_000,
                }
            },
        }
    )
    service = BillingService(subscriptions=subs, usage=_FakeUsage(), stripe=stripe)

    await service.handle_webhook(b"{}", "sig")

    assert subs.last_upsert is not None
    assert subs.last_upsert["user_id"] == "user-9"
    assert subs.last_upsert["plan"] == "free"
    assert subs.last_upsert["status"] == "canceled"


@pytest.mark.asyncio
async def test_free_quota_limit_rejected():
    subs = _FakeSubscriptions()
    stripe = _FakeStripe()

    class _FixedUsage(_FakeUsage):
        async def get_daily_usage(self, user_id: str, usage_date: date) -> DailyUsage | None:
            return DailyUsage(usage_date=usage_date, research_queries_count=3, total_questions_count=10)

    service = BillingService(subscriptions=subs, usage=_FixedUsage(), stripe=stripe)
    with pytest.raises(Exception) as exc:
        await service.check_and_consume_usage("u1", UsageIncrement(total_questions=1))
    assert "Daily question limit reached" in str(exc.value)


@pytest.mark.asyncio
async def test_pro_quota_allows_higher_usage():
    subs = _FakeSubscriptions()
    subs.by_user["u1"] = UserSubscription(user_id="u1", plan=Plan.PRO, status="active")
    stripe = _FakeStripe()

    class _HighUsage(_FakeUsage):
        async def get_daily_usage(self, user_id: str, usage_date: date) -> DailyUsage | None:
            return DailyUsage(usage_date=usage_date, research_queries_count=29, total_questions_count=99)

    service = BillingService(subscriptions=subs, usage=_HighUsage(), stripe=stripe)
    summary = await service.check_and_consume_usage("u1", UsageIncrement(research_queries=1, total_questions=1))
    assert summary.plan == Plan.PRO
    assert summary.usage.research_queries_count == 1
    assert summary.usage.total_questions_count == 1
