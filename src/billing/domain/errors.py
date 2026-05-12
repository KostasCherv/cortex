from __future__ import annotations

from dataclasses import dataclass


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
