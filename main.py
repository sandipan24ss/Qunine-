import os
import json
import re
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo
from typing import Optional, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Qunine demonstration backend
# Configure two separate Groq API keys as environment variables:
# GROQ_API_KEY_1 = understanding engine
# GROQ_API_KEY_2 = response engine

TZ = ZoneInfo("Asia/Kolkata")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
KEY_1 = os.getenv("GROQ_API_KEY_1", "")
KEY_2 = os.getenv("GROQ_API_KEY_2", "")

app = FastAPI(title="Qunine API", version="1.0.0")

origins = os.getenv("FRONTEND_ORIGINS", "*")
allowed_origins = [x.strip() for x in origins.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compact instructions for AI 1. Its only job is to interpret the question.
UNDERSTANDING_PROMPT = """
You are Qunine's understanding engine. Do not answer the user.
Interpret the latest user message using the conversation context.
Return ONLY valid JSON with these keys:
{
 "intent":"payments|general|product_info|how_it_works|capabilities|future_features|date_time|clarification",
 "date_range":"today|yesterday|all|unspecified",
 "person_name":null,
 "payment_app":null,
 "language":"Bengali|Hindi|English",
 "script":"roman|native|latin",
 "needs_payment_data":true,
 "clarification_question":null
}
Rules:
- Understand informal speech, spelling mistakes, mixed languages and short follow-ups.
- Romanized Bengali means Bengali written in English letters; reply script must be roman.
- Romanized Hindi means Hindi written in English letters; reply script must be roman.
- Bengali script means Bengali/native. Devanagari means Hindi/native. English means English/latin.
- Infer date range and person/app only when reasonably clear. Use null otherwise.
- "Ar Soumita?" refers to the prior payment question when context supports it.
- Questions about what Qunine is/how it works/current features/future features are not payment queries.
- If uncertain about the user's intent, use clarification.
- Do not invent records or answer the user.
"""

# Compact instructions for AI 2. It speaks using Python-verified facts.
RESPONSE_PROMPT = """
You are Qunine, a gentle and practical voice assistant for small shopkeepers and elderly users.
Answer the user's original question using the verified information supplied by Python.
Never invent payment records, amounts, names, dates, or calculations. Do not claim a payment is bank-verified unless the supplied data explicitly says so.
Qunine is currently a demonstration using simulated payment records. Actual Android notification capture from Google Pay, PhonePe, Paytm and other supported apps is under development. The intended process is: the user grants Android notification access, the app reads available payment notification details such as payer, amount, date and time, then sends them to the backend for storage and later search. Do not claim this is already working.
Future plans may include automatic payment tracking, searchable history, daily summaries, screen-guided assistance, smart reminders, and wider app/language support. Do not describe planned features as available.
If the user asks how Qunine works, explain the planned notification-capture process simply. If asked what is available, distinguish demo features from features under development and future plans.
If today's records are empty, answer directly that no payment has been recorded today. Do not suggest manual checking on the first answer. If the user repeats the question, doubts the answer, or insists someone paid, explain that this is a demonstration and actual notification detection is still under development. Never say the customer definitely did not pay.
Never request UPI PINs, OTPs, passwords, or banking credentials. Never claim to send or receive money.
Match the user's language and script. Romanized Bengali gets Romanized Bengali; Romanized Hindi gets Romanized Hindi. Bengali script gets Bengali script; Devanagari gets Hindi script; English gets English.
This response will be converted to speech. Use short, natural sentences. Avoid markdown, headings, lists, emojis, unnecessary symbols, and technical jargon. Be polite, patient, and concise.
"""

# These are simulated demonstration records, not real payments.
def make_demo_payments():
    now = datetime.now(TZ)
    yesterday = (now - timedelta(days=1)).date().isoformat()
    return [
        {
            "id": "demo_sandipan_yesterday",
            "payer_name": "Sandipan Roy",
            "amount": 1085,
            "currency": "INR",
            "app": "Google Pay",
            "date": yesterday,
            "time": "10:54 PM",
            "status": "simulated_demo",
            "source": "SIMULATED DEMO",
        },
        {
            "id": "demo_soumita_yesterday",
            "payer_name": "Soumita Roy",
            "amount": 1095,
            "currency": "INR",
            "app": "PhonePe",
            "date": yesterday,
            "time": "11:11 PM",
            "status": "simulated_demo",
            "source": "SIMULATED DEMO",
        },
    ]

PREVIOUS_PAYMENTS = make_demo_payments()
TODAY_PAYMENTS = []

# In-memory conversation sessions are suitable only for a small demo.
# On serverless hosting, use a persistent database for reliable session memory.
SESSIONS: dict[str, list[dict[str, str]]] = {}
MAX_HISTORY = 10


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    demo: bool = True


def local_now() -> datetime:
    return datetime.now(TZ)


def get_history(session_id: str) -> list[dict[str, str]]:
    return SESSIONS.get(session_id, [])[-MAX_HISTORY:]


def save_history(session_id: str, user_message: str, assistant_message: str):
    history = SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    SESSIONS[session_id] = history[-MAX_HISTORY:]


def infer_script_hint(message: str) -> str:
    if re.search(r"[\u0980-\u09FF]", message):
        return "Bengali script"
    if re.search(r"[\u0900-\u097F]", message):
        return "Hindi Devanagari script"
    return "Latin letters; determine whether the language is English or romanized Bengali/Hindi"


async def groq_chat(api_key: str, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="A required Groq API key is missing. Configure GROQ_API_KEY_1 and GROQ_API_KEY_2."
        )
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(GROQ_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            # Do not expose secret keys or full request headers.
            raise HTTPException(status_code=502, detail=f"Groq API error ({response.status_code}): {response.text[:500]}")
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq request timed out. Please try again.")
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not connect to the Groq service.")
    except (KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(status_code=502, detail="Unexpected response from the Groq service.")


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {
        "intent": "general",
        "date_range": "unspecified",
        "person_name": None,
        "payment_app": None,
        "language": "English",
        "script": "latin",
        "needs_payment_data": False,
        "clarification_question": None,
    }


def deterministic_date_fallback(message: str, interpretation: dict[str, Any]) -> dict[str, Any]:
    # Correct common date words if the model misses them.
    low = message.lower()
    if any(x in low for x in ["today", "today's", "ajke", "ajker", "aaj", "आज", "আজ"]):
        interpretation["date_range"] = "today"
    elif any(x in low for x in ["yesterday", "kalke", "kal er", "kal-er", "গতকাল", "कल"]):
        interpretation["date_range"] = "yesterday"
    return interpretation


def get_all_payments() -> list[dict[str, Any]]:
    return TODAY_PAYMENTS + PREVIOUS_PAYMENTS


def filter_payments(interpretation: dict[str, Any]) -> list[dict[str, Any]]:
    records = get_all_payments()
    date_range = interpretation.get("date_range", "unspecified")
    today = local_now().date()
    yesterday = today - timedelta(days=1)

    if date_range == "today":
        records = [p for p in records if p["date"] == today.isoformat()]
    elif date_range == "yesterday":
        records = [p for p in records if p["date"] == yesterday.isoformat()]

    person = interpretation.get("person_name")
    if isinstance(person, str) and person.strip():
        term = person.strip().lower()
        records = [p for p in records if term in p["payer_name"].lower()]

    app_name = interpretation.get("payment_app")
    if isinstance(app_name, str) and app_name.strip():
        term = app_name.strip().lower()
        records = [p for p in records if term in p["app"].lower()]

    return records


def build_payment_context(records: list[dict[str, Any]], interpretation: dict[str, Any]) -> dict[str, Any]:
    total = sum((Decimal(str(p["amount"])) for p in records), Decimal("0"))
    return {
        "records": records,
        "count": len(records),
        "total_amount": float(total),
        "currency": "INR",
        "all_records_are_simulated": all(p.get("status") == "simulated_demo" for p in records) if records else True,
        "requested_date_range": interpretation.get("date_range", "unspecified"),
        "today": local_now().date().isoformat(),
        "weekday": local_now().strftime("%A"),
        "current_time_ist": local_now().strftime("%I:%M %p"),
    }


def should_fetch_payments(interpretation: dict[str, Any]) -> bool:
    intent = interpretation.get("intent")
    return bool(interpretation.get("needs_payment_data")) or intent == "payments"


@app.get("/")
async def root():
    return {"name": "Qunine", "status": "running", "demo": True}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "understanding_key_configured": bool(KEY_1),
        "response_key_configured": bool(KEY_2),
        "demo": True,
    }


@app.get("/api/payments")
async def payments():
    return {
        "today": TODAY_PAYMENTS,
        "previous": PREVIOUS_PAYMENTS,
        "demo": True,
        "note": "These are simulated demonstration records, not real transactions."
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()
    session_id = request.session_id or str(uuid.uuid4())
    history = get_history(session_id)

    # CALL 1: Understanding engine — uses GROQ_API_KEY_1 only.
    understanding_messages = [
        {"role": "system", "content": UNDERSTANDING_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f"Writing-script hint: {infer_script_hint(message)}\n"
                f"Current date and time in India: {local_now().isoformat()}\n"
                f"Latest user message: {message}"
            ),
        },
    ]
    raw_interpretation = await groq_chat(KEY_1, understanding_messages, temperature=0.0)
    interpretation = deterministic_date_fallback(
        message, parse_json_object(raw_interpretation)
    )

    intent = interpretation.get("intent", "general")
    payment_context = {
        "records": [],
        "count": 0,
        "total_amount": 0,
        "currency": "INR",
        "all_records_are_simulated": True,
        "requested_date_range": interpretation.get("date_range", "unspecified"),
        "today": local_now().date().isoformat(),
        "weekday": local_now().strftime("%A"),
        "current_time_ist": local_now().strftime("%I:%M %p"),
    }

    if should_fetch_payments(interpretation):
        records = filter_payments(interpretation)
        payment_context = build_payment_context(records, interpretation)

    # CALL 2: Response engine — uses GROQ_API_KEY_2 only.
    response_payload = {
        "original_user_message": message,
        "understanding": interpretation,
        "verified_python_information": payment_context,
        "product_status": {
            "current_version": "demonstration",
            "current_features": [
                "AI conversation",
                "answers using Python-provided simulated payment records",
                "backend payment filtering and calculations",
            ],
            "under_development": [
                "Android notification reading",
                "automatic payment detail extraction and collection",
            ],
            "future_plans": [
                "persistent searchable payment history",
                "daily summaries",
                "screen-guided assistance",
                "smart reminders",
                "expanded payment app and language support",
            ],
        },
    }
    response_messages = [
        {"role": "system", "content": RESPONSE_PROMPT},
        *history,
        {
            "role": "user",
            "content": json.dumps(response_payload, ensure_ascii=False),
        },
    ]
    reply = await groq_chat(KEY_2, response_messages, temperature=0.45)
    reply = reply.strip().strip('"')

    if not reply:
        reply = "Sorry, I could not prepare an answer just now. Please try again."

    save_history(session_id, message, reply)
    return ChatResponse(reply=reply, session_id=session_id, demo=True)
