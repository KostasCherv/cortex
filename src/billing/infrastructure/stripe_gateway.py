from __future__ import annotations

import logging

import httpx
import stripe

from src.billing.application.ports import StripeGateway
from src.config import settings

logger = logging.getLogger(__name__)


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
        data = event.to_dict_recursive()
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

    def construct_webhook_event(self, payload: bytes, signature: str) -> dict:
        raise RuntimeError("Stripe webhook is not configured.")
