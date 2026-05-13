from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging

from src.billing.application.ports import StripeGateway, SubscriptionRepository, UsageRepository
from src.billing.domain.errors import BillingSyncError, QuotaExceededError
from src.billing.domain.models import DailyUsage, Plan, UsageSummary
from src.billing.domain.policy import limits_for_plan, next_utc_midnight, utc_today

logger = logging.getLogger(__name__)


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
