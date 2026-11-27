import os
import re
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Qunine backend — Vercel serverless compatible
TZ = ZoneInfo("Asia/Kolkata")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

yesterday = (datetime.now(TZ).date() - timedelta(days=1)).isoformat()

PREVIOUS_PAYMENTS = [
    {
        "name": "Sandipan Roy",
        "amount": 1085,
        "app": "Google Pay",
        "time": "10:54 PM",
        "date": yesterday,
        "period": "yesterday",
        "status": "SIMULATED DEMO — not a real or bank-verified payment",
    },
    {
        "name": "Soumita Roy",
        "amount": 1095,
        "app": "PhonePe",
        "time": "11:11 PM",
        "date": yesterday,
        "period": "yesterday",
        "status": "SIMULATED DEMO — not a real or bank-verified payment",
    },
]
TODAY_PAYMENTS = []  # No live payment integration is connected yet.

SYSTEM_PROMPT = """
You are Qunine, a polite, practical payment-record assistant for shopkeepers.

LANGUAGE: Reply in the language of the user's latest message. English in, English
out; Bengali in, Bengali out; Hindi in, Hindi out. Never switch languages randomly.

IDENTITY: Qunine is a demonstration version. It is an early prototype and has a
long way to go. Be honest about what works now and what is planned.

ROLE: Answer using only the payment records and app details provided in the
current request. Help find payments, explain amounts, dates, times and totals.
Never invent records or claim access to information not supplied.

CURRENT LIMITATIONS: Live phone notifications, installed-app detection, bank
connection and bank verification are not available in this demo. The sample
records mention Google Pay and PhonePe, but that does not prove either app is
installed on the user's phone. Do not claim to know which apps are installed.

MANUAL RECHECK: If a payment seems missing, first explain the records you have.
Then guide the user to open the payment app they actually use and check its
transaction history/activity. Ask which app they use if it is unknown; give
simple app-specific steps when known. Qunine cannot currently view or control
the user's screen or start screen sharing. Explain that screen-share guidance
is a planned capability but unavailable due to current limitations. Never ask
for a UPI PIN, OTP, password or banking credentials.

ACCURACY: Separate today's records from yesterday's. DEMO/SIMULATED records
are examples, not real payments. A notification is not bank confirmation.
If today's list is empty, say no payment notifications are recorded for today,
then offer to recheck together manually. Do not imply a payment probably exists.

STYLE: Answer first, politely and clearly. Keep ordinary spoken answers brief;
give step-by-step detail when guiding a manual check. Finish every sentence.
"""

app = FastAPI(title="Qunine API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("FRONTEND_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

def payment_context():
    return {
        "today_date": datetime.now(TZ).date().isoformat(),
        "today_payments": TODAY_PAYMENTS,
        "today_count": len(TODAY_PAYMENTS),
        "today_total": sum(p["amount"] for p in TODAY_PAYMENTS),
        "yesterday_payments": PREVIOUS_PAYMENTS,
        "yesterday_count": len(PREVIOUS_PAYMENTS),
        "yesterday_total": sum(p["amount"] for p in PREVIOUS_PAYMENTS),
        "note": "All previous payments are SIMULATED DEMO records, not real payments.",
    }

def is_today_payment_question(message: str) -> bool:
    m = message.lower()
    payment_words = ("payment", "payments", "paid", "received", "receive", "transaction", "money", "পেমেন্ট", "টাকা")
    today_words = ("today", "todays", "today's", "আজ", "আজকে")
    return any(w in m for w in payment_words) and any(w in m for w in today_words)

def is_yesterday_payment_question(message: str) -> bool:
    m = message.lower()
    return any(w in m for w in ("yesterday", "গতকাল")) and any(
        w in m for w in ("payment", "payments", "paid", "received", "transaction", "পেমেন্ট", "টাকা")
    )

def is_system_explanation(message: str) -> bool:
    m = message.lower()
    cues = ("how do you work", "how does your system work", "how does qunine work",
            "what does your system do", "what is your system", "how it works",
            "how do you collect", "what data do you collect", "কীভাবে কাজ")
    return any(x in m for x in cues)

def deterministic_answer(message: str):
    if is_today_payment_question(message):
        if not TODAY_PAYMENTS:
            return ("No payment notifications are recorded for today. If you think a payment "
                    "should be there, we can recheck it together manually in your payment "
                    "app's transaction history.")
        total = sum(p["amount"] for p in TODAY_PAYMENTS)
        return f"Yes. I have {len(TODAY_PAYMENTS)} payment records for today, totalling ₹{total:,}."

    if is_yesterday_payment_question(message):
        total = sum(p["amount"] for p in PREVIOUS_PAYMENTS)
        lines = [f"Yesterday has {len(PREVIOUS_PAYMENTS)} simulated demo payments, totalling ₹{total:,}."]
        for p in PREVIOUS_PAYMENTS:
            lines.append(f'{p["name"]}: ₹{p["amount"]:,} via {p["app"]} at {p["time"]}.')
        lines.append("These are demo examples, not real or bank-verified payments.")
        return " ".join(lines)

    if is_system_explanation(message):
        return ("Qunine is a demonstration version, and we still have a long way to go. "
                "The planned system will receive payment-notification details, such as the "
                "payer, amount, app, date and time, from a phone app that the shopkeeper "
                "explicitly permits. The backend will organize those records so Qunine can "
                "search them and answer questions. In this demo, live notifications, "
                "installed-app detection, bank connection and screen sharing are not active. "
                "Only two simulated examples from yesterday are loaded. If a payment seems "
                "missing, I can guide you to the transaction history in the payment app you "
                "use, but I cannot view your screen yet.")

    return None

async def ask_groq(message: str):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on Vercel.")
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "CURRENT PAYMENT DATA (JSON): " + json.dumps(payment_context(), ensure_ascii=False)},
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    try:
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Groq API error: {response.text[:500]}")
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            raise HTTPException(status_code=502, detail="The AI returned an empty response.")
        return answer
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Qunine took too long to respond. Please try again.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Could not get a response from Groq.")

@app.get("/")
async def root():
    return {"service": "Qunine API", "status": "online", "model": GROQ_MODEL}

@app.get("/api/health")
async def health():
    return {"status": "online", "model": GROQ_MODEL, "groq_key_configured": bool(GROQ_API_KEY)}

@app.get("/api/payments")
async def payments(period: str = "all"):
    if period == "today":
        return {"payments": TODAY_PAYMENTS, "count": len(TODAY_PAYMENTS), "total": sum(p["amount"] for p in TODAY_PAYMENTS)}
    if period == "yesterday":
        return {"payments": PREVIOUS_PAYMENTS, "count": len(PREVIOUS_PAYMENTS), "total": sum(p["amount"] for p in PREVIOUS_PAYMENTS)}
    if period == "all":
        records = TODAY_PAYMENTS + PREVIOUS_PAYMENTS
        return {"payments": records, "count": len(records), "total": sum(p["amount"] for p in records)}
    raise HTTPException(status_code=400, detail="period must be today, yesterday or all")

@app.post("/api/chat")
async def chat(data: ChatRequest):
    message = data.message.strip()
    answer = deterministic_answer(message)
    if answer is None:
        answer = await ask_groq(message)
    return {"reply": answer, "model": GROQ_MODEL}
