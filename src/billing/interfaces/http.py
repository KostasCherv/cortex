from __future__ import annotations

from dataclasses import asdict

from src.billing.application.service import BillingService
from src.billing.domain.models import UsageSummary


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
    from src.db.supabase_store import SupabaseSessionStore
    from src.billing.infrastructure.supabase_repositories import (
        SupabaseSubscriptionRepository,
        SupabaseUsageRepository,
    )
    from src.billing.infrastructure.stripe_gateway import NoopStripeGateway, StripeHttpGateway
    from src.config import settings

    store = SupabaseSessionStore()
    stripe_gateway = StripeHttpGateway() if settings.stripe_secret_key else NoopStripeGateway()
    return BillingService(
        subscriptions=SupabaseSubscriptionRepository(store),
        usage=SupabaseUsageRepository(store),
        stripe=stripe_gateway,
    )
