from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from .models import Plan, QuotaLimits


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
