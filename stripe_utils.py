"""
Stripe helpers: checkout session creation, webhook handling.

Environment variables required:
  STRIPE_SECRET_KEY        - your Stripe secret key (sk_live_... or sk_test_...)
  STRIPE_WEBHOOK_SECRET    - webhook signing secret (whsec_...)
  STRIPE_PRICE_ID          - price ID for the $9/month Pro plan (price_...)
  APP_URL                  - your public URL e.g. https://darwincards-production.up.railway.app
"""

import os
import stripe
from fastapi import HTTPException

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID       = os.environ.get("STRIPE_PRICE_ID", "")
APP_URL               = os.environ.get("APP_URL", "http://localhost:8000")


def create_checkout_session(user_email: str, user_id: int) -> str:
    """Creates a Stripe Checkout session and returns the URL."""
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode="subscription",
        customer_email=user_email,
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=f"{APP_URL}/?upgraded=1",
        cancel_url=f"{APP_URL}/?upgraded=0",
        metadata={"user_id": str(user_id)},
        subscription_data={"metadata": {"user_id": str(user_id)}},
    )
    return session.url


def create_portal_session(stripe_customer_id: str) -> str:
    """Creates a Stripe Customer Portal session (to manage/cancel subscription)."""
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=f"{APP_URL}/",
    )
    return session.url


def handle_webhook(payload: bytes, sig_header: str):
    """
    Processes Stripe webhook events.
    Returns (event_type, user_id, subscription_id, customer_id) or raises.
    """
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    return event
