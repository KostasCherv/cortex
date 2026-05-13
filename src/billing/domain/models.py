from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


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
