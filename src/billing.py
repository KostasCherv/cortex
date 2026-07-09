"""Billing: plans, quotas, Stripe subscriptions (flattened from src/billing/ package)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import Protocol

import httpx
import stripe

from src.config import settings
from src.db.supabase_store import SupabaseSessionStore

logger = logging.getLogger(__name__)


# --- errors ---


@dataclass
class QuotaExceededError(Exception):
    plan: str
    limit_type: str
    limit: int
    used: int
    resets_at: str
    message: str

    def __str__(self) -> str:
        return self.message


class BillingSyncError(Exception):
    pass


# --- models ---


class Plan(StrEnum):
    FREE = "free"
    PRO = "pro"


@dataclass(frozen=True)
class QuotaLimits:
    research_queries_daily: int
    total_questions_daily: int


@dataclass(frozen=True)
class DailyUsage:
    usage_date: date
    research_queries_count: int
    total_questions_count: int


@dataclass(frozen=True)
class UserSubscription:
    user_id: str
    plan: Plan
    status: str
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool | None = None
    cancel_at: datetime | None = None
    canceled_at: datetime | None = None


@dataclass(frozen=True)
class UsageSummary:
    plan: Plan
    date: date
    limits: QuotaLimits
    usage: DailyUsage
    resets_at: datetime
    subscription: UserSubscription | None = None


# --- policy ---


def limits_for_plan(plan: Plan) -> QuotaLimits:
    if plan == Plan.PRO:
        return QuotaLimits(research_queries_daily=30, total_questions_daily=100)
    return QuotaLimits(research_queries_daily=3, total_questions_daily=10)


def utc_today(now: datetime | None = None) -> date:
    current = now or datetime.now(UTC)
    return current.astimezone(UTC).date()


def next_utc_midnight(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    today = current.astimezone(UTC).date()
    tomorrow = today + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=UTC)


# --- ports ---


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

    async def get_subscription(self, subscription_id: str) -> dict | None: ...

    def construct_webhook_event(self, payload: bytes, signature: str) -> dict: ...


# --- service ---


@dataclass(frozen=True)
class UsageIncrement:
    research_queries: int = 0
    total_questions: int = 0


class BillingService:
    def __init__(
        self,
        *,
        subscriptions: SubscriptionRepository,
        usage: UsageRepository,
        stripe: StripeGateway,
    ) -> None:
        self._subscriptions = subscriptions
        self._usage = usage
        self._stripe = stripe

    async def get_effective_plan(self, user_id: str) -> Plan:
        record = await self._subscriptions.get_user_subscription(user_id)
        if record and record.status in {"active", "trialing"}:
            return record.plan
        return Plan.FREE

    async def get_usage_summary(self, user_id: str) -> UsageSummary:
        subscription = await self._subscriptions.get_user_subscription(user_id)
        plan = (
            subscription.plan
            if subscription and subscription.status in {"active", "trialing"}
            else Plan.FREE
        )
        limits = limits_for_plan(plan)
        usage_date = utc_today()
        usage = await self._usage.get_daily_usage(user_id, usage_date)
        if usage is None:
            usage = DailyUsage(usage_date=usage_date, research_queries_count=0, total_questions_count=0)
        return UsageSummary(
            plan=plan,
            date=usage_date,
            limits=limits,
            usage=usage,
            resets_at=next_utc_midnight(),
            subscription=subscription,
        )

    async def check_and_consume_usage(self, user_id: str, increment: UsageIncrement) -> UsageSummary:
        summary = await self.get_usage_summary(user_id)
        limits = summary.limits
        usage = summary.usage

        projected_research = usage.research_queries_count + increment.research_queries
        projected_questions = usage.total_questions_count + increment.total_questions

        if projected_research > limits.research_queries_daily:
            raise QuotaExceededError(
                plan=summary.plan.value,
                limit_type="research_daily",
                limit=limits.research_queries_daily,
                used=usage.research_queries_count,
                resets_at=summary.resets_at.isoformat(),
                message="Daily research query limit reached.",
            )

        if projected_questions > limits.total_questions_daily:
            raise QuotaExceededError(
                plan=summary.plan.value,
                limit_type="questions_daily",
                limit=limits.total_questions_daily,
                used=usage.total_questions_count,
                resets_at=summary.resets_at.isoformat(),
                message="Daily question limit reached.",
            )

        updated_usage = await self._usage.increment_daily_usage(
            user_id=user_id,
            usage_date=summary.date,
            add_research_queries=increment.research_queries,
            add_total_questions=increment.total_questions,
        )

        return UsageSummary(
            plan=summary.plan,
            date=summary.date,
            limits=limits,
            usage=updated_usage,
            resets_at=summary.resets_at,
        )

    async def start_checkout(self, *, user_id: str, email: str | None) -> str:
        return await self._stripe.create_checkout_session(user_id=user_id, email=email)

    async def start_portal(self, *, user_id: str) -> str:
        sub = await self._subscriptions.get_user_subscription(user_id)
        customer_id = sub.stripe_customer_id if sub else None
        if not customer_id:
            raise BillingSyncError("No Stripe customer found for this user.")
        return await self._stripe.create_portal_session(customer_id=customer_id)

    async def handle_webhook(self, payload: bytes, signature: str) -> None:
        event = self._stripe.construct_webhook_event(payload, signature)
        event_type = str(event.get("type", ""))
        data = event.get("data", {}).get("object", {})
        user_id = await self._resolve_user_id_for_event(event_type, data)

        if event_type == "checkout.session.completed":
            if not user_id:
                logger.warning("[billing] checkout.session.completed missing user_id; ignoring.")
                return
            subscription_id = data.get("subscription")
            customer_id = data.get("customer")
            status = "active"
            current_period_start = None
            current_period_end = None
            cancel_at_period_end = None
            cancel_at = None
            canceled_at = None
            if isinstance(subscription_id, str) and subscription_id:
                try:
                    subscription = await self._stripe.get_subscription(subscription_id)
                except Exception as exc:
                    logger.warning(
                        "[billing] failed to fetch subscription details for checkout session. "
                        "subscription_id=%s error=%s",
                        subscription_id,
                        exc,
                    )
                    subscription = None
                if isinstance(subscription, dict):
                    subscription_status = subscription.get("status")
                    if isinstance(subscription_status, str) and subscription_status:
                        status = subscription_status
                    current_period_start = _subscription_unix_to_iso(
                        subscription,
                        "current_period_start",
                    )
                    current_period_end = _subscription_unix_to_iso(
                        subscription,
                        "current_period_end",
                    )
                    cancel_at_period_end = subscription.get("cancel_at_period_end")
                    cancel_at = _unix_to_iso(subscription.get("cancel_at"))
                    canceled_at = _unix_to_iso(subscription.get("canceled_at"))
                    maybe_customer = subscription.get("customer")
                    if isinstance(maybe_customer, str) and maybe_customer:
                        customer_id = maybe_customer
            await self._subscriptions.upsert_user_subscription(
                {
                    "user_id": user_id,
                    "plan": "pro",
                    "status": status,
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": subscription_id,
                    "current_period_start": current_period_start,
                    "current_period_end": current_period_end,
                    "cancel_at_period_end": cancel_at_period_end,
                    "cancel_at": cancel_at,
                    "canceled_at": canceled_at,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            return

        if event_type.startswith("customer.subscription."):
            if not user_id:
                logger.warning("[billing] subscription event missing user_id; ignoring. type=%s", event_type)
                return
            status = data.get("status") or "incomplete"
            plan = "pro" if status in {"active", "trialing", "past_due"} else "free"
            await self._subscriptions.upsert_user_subscription(
                {
                    "user_id": user_id,
                    "plan": plan,
                    "status": status,
                    "stripe_customer_id": data.get("customer"),
                    "stripe_subscription_id": data.get("id"),
                    "current_period_start": _subscription_unix_to_iso(data, "current_period_start"),
                    "current_period_end": _subscription_unix_to_iso(data, "current_period_end"),
                    "cancel_at_period_end": data.get("cancel_at_period_end"),
                    "cancel_at": _unix_to_iso(data.get("cancel_at")),
                    "canceled_at": _unix_to_iso(data.get("canceled_at")),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            return

        if event_type.startswith("customer.subscription.") or event_type.startswith("checkout.session."):
            logger.info("[billing] webhook type %s ignored by handler.", event_type)

    async def _resolve_user_id_for_event(self, event_type: str, data: dict) -> str | None:
        metadata = data.get("metadata") or {}
        user_id = metadata.get("user_id")
        if isinstance(user_id, str) and user_id:
            return user_id

        if event_type.startswith("customer.subscription."):
            subscription_id = data.get("id")
            if isinstance(subscription_id, str) and subscription_id:
                record = await self._subscriptions.get_user_subscription_by_subscription_id(subscription_id)
                if record:
                    return record.user_id
            customer_id = data.get("customer")
            if isinstance(customer_id, str) and customer_id:
                record = await self._subscriptions.get_user_subscription_by_customer_id(customer_id)
                if record:
                    return record.user_id

        if event_type == "checkout.session.completed":
            client_reference_id = data.get("client_reference_id")
            if isinstance(client_reference_id, str) and client_reference_id:
                return client_reference_id
            customer_id = data.get("customer")
            if isinstance(customer_id, str) and customer_id:
                record = await self._subscriptions.get_user_subscription_by_customer_id(customer_id)
                if record:
                    return record.user_id

        return None


def _unix_to_iso(value: object) -> str | None:
    if not isinstance(value, int):
        return None
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _subscription_unix_to_iso(subscription: dict, key: str) -> str | None:
    """Read subscription period fields across Stripe API versions.

    Newer versions may expose current_period_* on subscription items rather than top-level.
    """
    top_level = _unix_to_iso(subscription.get(key))
    if top_level:
        return top_level
    items = subscription.get("items", {}).get("data", [])
    if not isinstance(items, list) or not items:
        return None
    first_item = items[0]
    if not isinstance(first_item, dict):
        return None
    return _unix_to_iso(first_item.get(key))


# --- stripe gateway ---


class StripeHttpGateway(StripeGateway):
    def __init__(self) -> None:
        if not settings.stripe_secret_key:
            raise RuntimeError("Stripe is not configured.")
        self._secret_key = settings.stripe_secret_key
        stripe.api_key = settings.stripe_secret_key

    async def create_checkout_session(self, *, user_id: str, email: str | None) -> str:
        if not settings.stripe_pro_price_id:
            raise RuntimeError("STRIPE_PRO_PRICE_ID is not configured.")
        payload = {
            "mode": "subscription",
            "success_url": settings.stripe_success_url,
            "cancel_url": settings.stripe_cancel_url,
            "line_items[0][price]": settings.stripe_pro_price_id,
            "line_items[0][quantity]": "1",
            "client_reference_id": user_id,
            "metadata[user_id]": user_id,
            "subscription_data[metadata][user_id]": user_id,
        }
        if email:
            payload["customer_email"] = email

        data = await self._stripe_post("/v1/checkout/sessions", payload)
        url = data.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("Stripe Checkout session URL missing.")
        return url

    async def create_portal_session(self, *, customer_id: str) -> str:
        payload = {
            "customer": customer_id,
            "return_url": settings.stripe_portal_return_url or settings.stripe_success_url,
        }
        data = await self._stripe_post("/v1/billing_portal/sessions", payload)
        url = data.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("Stripe portal session URL missing.")
        return url

    async def get_subscription(self, subscription_id: str) -> dict | None:
        if not subscription_id.strip():
            return None
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"https://api.stripe.com/v1/subscriptions/{subscription_id}",
                headers={"Authorization": f"Bearer {self._secret_key}"},
            )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            logger.warning(
                "[billing] Stripe subscription fetch failed: %s %s",
                response.status_code,
                response.text,
            )
            raise RuntimeError("Stripe subscription fetch failed.")
        data = response.json()
        return data if isinstance(data, dict) else None

    def construct_webhook_event(self, payload: bytes, signature: str) -> dict:
        if not settings.stripe_webhook_secret:
            raise RuntimeError("STRIPE_WEBHOOK_SECRET is not configured.")
        if not signature.strip():
            raise RuntimeError("Missing Stripe signature header.")
        try:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature,
                secret=settings.stripe_webhook_secret,
            )
        except Exception as exc:
            raise RuntimeError("Invalid Stripe webhook signature or payload.") from exc
        data = _stripe_event_to_dict(event)
        if not isinstance(data, dict) or "type" not in data:
            raise RuntimeError("Malformed Stripe webhook event.")
        return data

    async def _stripe_post(self, path: str, form_payload: dict[str, str]) -> dict:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"https://api.stripe.com{path}",
                headers={"Authorization": f"Bearer {self._secret_key}"},
                data=form_payload,
            )
        if response.status_code >= 400:
            logger.warning("[billing] Stripe request failed: %s %s", response.status_code, response.text)
            raise RuntimeError("Stripe request failed.")
        return response.json()


class NoopStripeGateway(StripeGateway):
    async def create_checkout_session(self, *, user_id: str, email: str | None) -> str:
        raise RuntimeError("Stripe checkout is not configured.")

    async def create_portal_session(self, *, customer_id: str) -> str:
        raise RuntimeError("Stripe portal is not configured.")

    async def get_subscription(self, subscription_id: str) -> dict | None:
        raise RuntimeError("Stripe subscription fetch is not configured.")

    def construct_webhook_event(self, payload: bytes, signature: str) -> dict:
        raise RuntimeError("Stripe webhook is not configured.")


def _stripe_event_to_dict(event: object) -> dict:
    """Convert Stripe SDK event objects to plain dicts across SDK versions."""
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return data

    to_dict_recursive = getattr(event, "to_dict_recursive", None)
    if callable(to_dict_recursive):
        data = to_dict_recursive()
        if isinstance(data, dict):
            return data

    internal_recursive = getattr(event, "_to_dict_recursive", None)
    if callable(internal_recursive):
        data = internal_recursive()
        if isinstance(data, dict):
            return data

    if isinstance(event, dict):
        return event

    raise RuntimeError("Malformed Stripe webhook event.")


# --- supabase repositories ---


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


# --- http interface ---


def usage_summary_to_response(summary: UsageSummary) -> dict:
    subscription = summary.subscription
    return {
        "plan": summary.plan.value,
        "date": summary.date.isoformat(),
        "limits": asdict(summary.limits),
        "usage": {
            "research_queries_count": summary.usage.research_queries_count,
            "total_questions_count": summary.usage.total_questions_count,
        },
        "resets_at": summary.resets_at.isoformat(),
        "subscription": (
            {
                "status": subscription.status,
                "current_period_end": (
                    subscription.current_period_end.isoformat()
                    if subscription.current_period_end
                    else None
                ),
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "cancel_at": subscription.cancel_at.isoformat() if subscription.cancel_at else None,
                "canceled_at": subscription.canceled_at.isoformat() if subscription.canceled_at else None,
            }
            if subscription
            else None
        ),
    }


def build_billing_service() -> BillingService:
    store = SupabaseSessionStore()
    stripe_gateway = StripeHttpGateway() if settings.stripe_secret_key else NoopStripeGateway()
    return BillingService(
        subscriptions=SupabaseSubscriptionRepository(store),
        usage=SupabaseUsageRepository(store),
        stripe=stripe_gateway,
    )
