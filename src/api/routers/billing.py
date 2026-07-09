"""Billing routes: usage, Stripe checkout/portal, and webhook."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.deps import _get_billing_service
from src.auth import AuthenticatedUser, get_authenticated_user
from src.billing import BillingSyncError, usage_summary_to_response

router = APIRouter()


class BillingCheckoutRequest(BaseModel):
    pass


@router.get("/api/billing/usage", tags=["Billing"])
async def billing_usage(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    summary = await _get_billing_service().get_usage_summary(current_user.user_id)
    return usage_summary_to_response(summary)


@router.post("/api/billing/checkout-session", tags=["Billing"])
async def create_checkout_session(
    _body: BillingCheckoutRequest,
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        checkout_url = await _get_billing_service().start_checkout(
            user_id=current_user.user_id,
            email=current_user.email,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"url": checkout_url}


@router.post("/api/billing/portal-session", tags=["Billing"])
async def create_portal_session(
    current_user: AuthenticatedUser = Depends(get_authenticated_user),
):
    try:
        portal_url = await _get_billing_service().start_portal(
            user_id=current_user.user_id
        )
    except BillingSyncError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"url": portal_url}


@router.post("/api/billing/webhook", tags=["Billing"])
async def stripe_webhook(request: Request):
    signature = request.headers.get("Stripe-Signature", "")
    payload = await request.body()
    try:
        await _get_billing_service().handle_webhook(payload, signature)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse({"received": True})
