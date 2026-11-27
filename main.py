"""
Qunine backend
---------------
Install:
    pip install fastapi uvicorn groq python-dotenv

Set your Groq API key as an environment variable:
    GROQ_API_KEY=your_key_here

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

This prototype contains DEMO payment records only.
It does not read real payment-app notifications.
"""

import os
import re
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

APP_NAME = "Qunine API"
TIMEZONE = ZoneInfo("Asia/Kolkata")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(
    title=APP_NAME,
    description="Backend API for the Qunine voice business assistant.",
    version="1.0.0",
)

# Prototype setting. Restrict allow_origins to your real frontend domain
# before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep this prompt short to reduce repeated input-token usage.
SYSTEM_PROMPT = """
You are Qunine, a gentle, patient and polite business assistant for shopkeepers.
Understand Bengali, Hindi, English and mixed-language speech. Reply in the
user's language using simple, natural, concise sentences suitable for speaking.
Use only the transaction data supplied in the current request. Never invent
names, amounts, dates, times or statuses. A notification is not proof that
money reached a bank account: distinguish detected, confirmed and unverified
payments. All records marked DEMO or SIMULATED are examples, never real money.
If a payment notification is missing, explain that it may be delayed or missed
and guide the user to check the payment app's transaction history and, if needed,
their bank's official record. Never ask for a UPI PIN, OTP or password. Never
claim to send, receive or verify money unless the supplied data proves it.
Keep answers short and friendly. If information is unavailable, say so.
"""

# ---------------------------------------------------------
# DEMO PAYMENT DATA
# ---------------------------------------------------------

def india_now() -> datetime:
    return datetime.now(TIMEZONE)


def yesterday_at(hour: int, minute: int) -> datetime:
    yesterday = india_now().date() - timedelta(days=1)
    return datetime.combine(
        yesterday,
        time(hour=hour, minute=minute),
        tzinfo=TIMEZONE,
    )


# These two records are intentionally labelled as yesterday's
# simulated demo payments. They are not real or verified payments.
PREVIOUS_PAYMENTS = [
    {
        "id": "demo-yesterday-001",
        "name": "Sandipan Roy",
        "amount": 1085,
        "currency": "INR",
        "app": "Google Pay",
        "time": "10:54 PM",
        "timestamp": yesterday_at(22, 54).isoformat(),
        "date": (india_now().date() - timedelta(days=1)).isoformat(),
        "status": "SIMULATED",
        "type": "DEMO",
        "period": "YESTERDAY",
    },
    {
        "id": "demo-yesterday-002",
        "name": "Soumita Roy",
        "amount": 1095,
        "currency": "INR",
        "app": "PhonePe",
        "time": "11:11 PM",
        "timestamp": yesterday_at(23, 11).isoformat(),
        "date": (india_now().date() - timedelta(days=1)).isoformat(),
        "status": "SIMULATED",
        "type": "DEMO",
        "period": "YESTERDAY",
    },
]

# Today's list starts empty. Add only records supplied by a trusted
# backend integration. Do not treat a notification alone as confirmed.
TODAY_PAYMENTS = []


# ---------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    source: str
    payments: list = []


# ---------------------------------------------------------
# PAYMENT HELPERS
# ---------------------------------------------------------

def money(amount: int) -> str:
    return f"₹{amount:,}"


def payment_summary(records: list) -> dict:
    return {
        "count": len(records),
        "total": sum(int(p["amount"]) for p in records),
        "payments": records,
    }


def period_records(period: str) -> list:
    period = period.lower()
    if period in ("yesterday", "previous", "previous_day"):
        return PREVIOUS_PAYMENTS
    if period in ("today", "current"):
        return TODAY_PAYMENTS
    return PREVIOUS_PAYMENTS + TODAY_PAYMENTS


def make_payment_details(records: list) -> str:
    if not records:
        return "There are no payment records for that period."

    lines = []
    for p in records:
        lines.append(
            f'{p["name"]}: {money(p["amount"])} via {p["app"]} '
            f'at {p["time"]}. Status: {p["status"]}. Type: {p["type"]}.'
        )
    return " ".join(lines)


def local_intent_reply(message: str) -> Optional[ChatResponse]:
    """
    Answer straightforward payment questions with Python.
    This avoids calling Groq for routine totals and record lookups.
    """
    q = message.lower().strip()

    yesterday_words = ("yesterday", "previous", "last day")
    today_words = ("today", "this day")
    asks_yesterday = any(word in q for word in yesterday_words)
    asks_today = any(word in q for word in today_words)

    # Greetings: no Groq call needed.
    if re.fullmatch(
        r"(hi|hello|hey|hello qunine|hi qunine|good morning|good evening)[!. ]*",
        q,
    ):
        return ChatResponse(
            reply="Hello! I'm Qunine, your business assistant. How can I help you today?",
            source="python",
            payments=[],
        )

    # Pick the requested period. If no period is mentioned, use today
    # for today's summary questions.
    if asks_yesterday:
        records = PREVIOUS_PAYMENTS
        period_label = "yesterday's previous demo payments"
    elif asks_today:
        records = TODAY_PAYMENTS
        period_label = "today's payments"
    else:
        records = None
        period_label = ""

    asks_total = any(
        word in q for word in ("total", "how much", "amount", "kitna", "कितना", "মোট")
    )
    asks_count = any(
        word in q for word in ("how many", "number of", "count", "কটা", "কতগুলো")
    )
    asks_details = any(
        word in q for word in (
            "details", "detail", "all payments", "who paid", "names",
            "list", "show payments", "payment history", "সব পেমেন্ট", "নামগুলো"
        )
    )

    if records is not None and (asks_total or asks_count or asks_details):
        summary = payment_summary(records)

        if not records:
            reply = (
                "I don't have any payment records for today yet. "
                "I can only report payments that have been provided to me."
            )
        elif asks_details:
            reply = (
                f"These are {period_label}. There are {summary['count']} "
                f"simulated demo payments totalling {money(summary['total'])}. "
                + make_payment_details(records)
            )
        elif asks_count and not asks_total:
            reply = (
                f"You have {summary['count']} simulated demo payments in "
                f"{period_label}, totalling {money(summary['total'])}."
            )
        else:
            reply = (
                f"The total for {period_label} is {money(summary['total'])} "
                f"across {summary['count']} simulated demo payments."
            )

        return ChatResponse(
            reply=reply,
            source="python",
            payments=records,
        )

    # Exact customer lookup in the available demo records.
    all_records = PREVIOUS_PAYMENTS + TODAY_PAYMENTS
    for p in all_records:
        customer = p["name"].lower()
        if customer in q:
            return ChatResponse(
                reply=(
                    f"I found a {p['type']} record for {p['name']}: "
                    f"{money(p['amount'])} via {p['app']} at {p['time']}. "
                    f"It is marked {p['status']}, so it is a simulated demo "
                    f"record, not a confirmed real payment."
                ),
                source="python",
                payments=[p],
            )

    return None


# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "app": APP_NAME,
        "status": "running",
        "message": "Qunine backend is ready.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_configured": bool(groq_client),
        "model": GROQ_MODEL,
        "timezone": "Asia/Kolkata",
    }


@app.get("/api/payments")
def get_payments(period: str = "all"):
    """
    Examples:
      /api/payments?period=today
      /api/payments?period=yesterday
      /api/payments?period=all
    """
    if period.lower() in ("today", "current"):
        records = TODAY_PAYMENTS
    elif period.lower() in ("yesterday", "previous"):
        records = PREVIOUS_PAYMENTS
    elif period.lower() == "all":
        records = TODAY_PAYMENTS + PREVIOUS_PAYMENTS
    else:
        raise HTTPException(
            status_code=400,
            detail="period must be today, yesterday or all",
        )

    summary = payment_summary(records)
    return {
        "period": period.lower(),
        "currency": "INR",
        "count": summary["count"],
        "total": summary["total"],
        "payments": records,
        "demo_only": True,
    }


@app.get("/api/summary")
def get_summary():
    today_summary = payment_summary(TODAY_PAYMENTS)
    yesterday_summary = payment_summary(PREVIOUS_PAYMENTS)

    return {
        "today": {
            "count": today_summary["count"],
            "total": today_summary["total"],
            "payments": TODAY_PAYMENTS,
        },
        "yesterday": {
            "count": yesterday_summary["count"],
            "total": yesterday_summary["total"],
            "payments": PREVIOUS_PAYMENTS,
        },
        "currency": "INR",
        "note": "The supplied previous payments are simulated demo records.",
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()

    # First use Python for greetings, totals and exact record lookups.
    quick_reply = local_intent_reply(message)
    if quick_reply:
        return quick_reply

    # Use Groq only for questions needing flexible language understanding.
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail=(
                "Groq API key is not configured. Set GROQ_API_KEY "
                "in your environment or .env file."
            ),
        )

    # Send only the records relevant to the current conversation.
    # Both lists are demo data in this prototype.
    payment_context = {
        "today": TODAY_PAYMENTS,
        "yesterday": PREVIOUS_PAYMENTS,
        "currency": "INR",
        "note": (
            "All records in this prototype are simulated demo records. "
            "They are not real payments and are not independently verified."
        ),
    }

    user_content = (
        f"Available payment data (JSON): {payment_context}\n\n"
        f"Shopkeeper's message: {message}\n\n"
        "Answer naturally and briefly. Use only the supplied data. "
        "If the user asks about a payment not present in the data, say "
        "you cannot find it. Do not invent records."
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=180,
        )

        answer = completion.choices[0].message.content or (
            "I'm sorry, I couldn't prepare an answer just now. Please try again."
        )

        return ChatResponse(
            reply=answer.strip(),
            source="groq",
            payments=[],
        )

    except Exception as exc:
        # Avoid returning API keys or internal credentials to the frontend.
        raise HTTPException(
            status_code=502,
            detail=f"AI service request failed: {type(exc).__name__}",
        )


# Run with:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
