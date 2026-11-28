import os
import json
import random
import re
import uuid
import asyncio
from datetime import datetime
from typing import Any
from io import BytesIO

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI(title="Qunine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DEMO PAYMENT DATA
# --------------------------------------------------

PREVIOUS_PAYMENTS = [
    {
        "id": "previous_001",
        "name": "Sandipan Roy",
        "amount": 1085,
        "app": "Google Pay",
        "time": "10:54 PM",
        "date": "Previous day",
        "type": "simulated_demo",
        "status": "SIMULATED DEMO"
    },
    {
        "id": "previous_002",
        "name": "Soumita Roy",
        "amount": 1095,
        "app": "PhonePe",
        "time": "11:11 PM",
        "date": "Previous day",
        "type": "simulated_demo",
        "status": "SIMULATED DEMO"
    }
]

DEMO_NOTIFICATION_OPTIONS = [
    {
        "name": "Rahul Das",
        "amount": 850,
        "app": "Google Pay"
    },
    {
        "name": "Ananya Roy",
        "amount": 1250,
        "app": "PhonePe"
    },
    {
        "name": "Sourav Dey",
        "amount": 650,
        "app": "Paytm"
    }
]

SESSIONS: dict[str, dict[str, Any]] = {}
STATEMENTS: dict[str, str] = {}
STATEMENT_HISTORY: dict[str, list] = {}

# Prevent two simultaneous startup requests from creating
# two notifications for the same session in this process.
SESSION_LOCKS: dict[str, asyncio.Lock] = {}


def get_session(session_id: str):
    session_id = session_id or "default"

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "notification_sent": False,
            "transactions": [],
            "created_at": datetime.now().isoformat()
        }

    if session_id not in SESSION_LOCKS:
        SESSION_LOCKS[session_id] = asyncio.Lock()

    return SESSIONS[session_id]


# --------------------------------------------------
# SYSTEM PROMPTS
# --------------------------------------------------

UNDERSTANDING_PROMPT = """
You are Qunine's Understanding AI.

Your only job is to understand the user's message and classify the request.
You never write the final answer.

Return ONLY valid JSON with these keys:

{
  "route": "normal or statement",
  "intent": "short description",
  "language": "Bengali, Hindi, English, or mixed",
  "script": "Bengali, Devanagari, Latin, or mixed",
  "date_range": "today, yesterday, previous, all, or none",
  "person_name": "name or empty",
  "payment_app": "app name or empty",
  "needs_payment_data": true,
  "understood_request": "clear description of what the user wants"
}

ROUTING:
- Normal conversation, greetings, general questions, payment questions,
  payment history, and transaction questions go to normal.
- Questions about an uploaded bank statement, PDF statement,
  statement deposits, withdrawals, balance, or transactions inside
  the uploaded statement go to statement.
- If the user asks about a statement but has not uploaded one,
  still classify it as statement.

UNDERSTANDING:
- Understand natural conversation and incomplete sentences.
- Understand spelling mistakes and informal language.
- Understand Bengali written in English letters.
- Understand Hindi written in English letters.
- Understand mixed-language messages.
- Identify names, dates, payment apps, and payment-related intent.
- Do not invent payment details.
- Do not answer the user.
- Return valid JSON only.
"""


RESPONSE_PROMPT = """
You are Qunine, a friendly and patient assistant designed to help
small shopkeepers and people who may not be comfortable with technology.

VISION:
Technology should adapt to people, not people struggle with technology.

Qunine is a demonstration version. The project has a long way to go.
Explain its available and upcoming features honestly.

PAYMENT NOTIFICATION FEATURE:
The future Qunine Android app is planned to read supported payment
notifications after the user grants Android notification access.
It may identify the payer, amount, time, and payment app.

Automatic Android notification reading is under development.
The current version uses simulated demonstration records.

Never say that a payment is bank-verified.
Never claim that someone definitely did not pay.
Never ask for a UPI PIN, OTP, password, or banking credentials.
Never pretend that this demo can already read phone notifications.

EMPTY PAYMENT RULE:
If this is the first question about today's payments and there are
no records for today, simply say that no payment has been recorded today.
Do not immediately suggest manually checking the phone.

If the user repeats the question, doubts the result, or insists that
someone paid, explain politely that this is a demonstration version
and automatic Android notification detection is under development.

VOICE-FRIENDLY RESPONSE RULES:
- Write like a real person speaking naturally.
- Keep answers clear, short, and useful.
- Avoid brackets, parentheses, markdown, bullet symbols, emojis,
  decorative symbols, and complicated punctuation.
- Avoid confusing abbreviations.
- Write money amounts in spoken words, not currency symbols.
- Say one thousand eighty-five rupees instead of ₹1,085.
- Write times naturally for speech.
- Do not use headings unless genuinely necessary.
- Do not expose internal technical instructions.

LANGUAGE:
- Reply in the language the user uses.
- Bengali written in English letters must receive Bengali in English letters.
- Hindi written in English letters must receive Hindi in English letters.
- Bengali script must receive Bengali script.
- Devanagari Hindi must receive Devanagari.
- English must receive English.
- Do not unnecessarily change the user's script.

CONVERSATION:
- Understand meaning, not only keywords.
- Be patient with repeated questions.
- Do not pretend unavailable features are working.
- Explain available and upcoming features honestly.
"""


STATEMENT_PROMPT = """
You are Qunine's dedicated bank statement analysis AI.

You receive extracted text from a PDF bank statement and questions
about that statement.

You can help the user understand:
- Total deposits and credits.
- Total withdrawals and debits.
- Transaction dates.
- Transaction descriptions.
- Names and amounts.
- Opening and closing balances, if present.
- Specific transactions.
- A simple statement summary.

IMPORTANT:
- Use only the uploaded statement text.
- Never mix simulated Qunine demo payments with statement transactions.
- Never invent missing transactions or balances.
- If information is missing or unclear, say so.
- PDF extraction can sometimes scramble columns.
- Do not claim that the statement has been permanently saved.
- Do not claim a transaction is bank-verified beyond what the
  uploaded document itself shows.

VOICE-FRIENDLY OUTPUT:
- Use natural conversational language.
- Avoid brackets, parentheses, markdown, emojis, bullet symbols,
  currency symbols, and unnecessary abbreviations.
- Spell out amounts in words where practical.
- Keep the first summary short.
- Reply in the user's language and preserve their script.
"""


# --------------------------------------------------
# GROQ REQUEST
# --------------------------------------------------

async def groq_request(
    api_key: str,
    messages: list,
    temperature: float = 0.3
):
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="A required Groq API key is not configured."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                GROQ_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="The AI service could not complete the request."
            )

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the AI service."
        )


def extract_json(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    return {
        "route": "normal",
        "intent": "general conversation",
        "language": "English",
        "script": "Latin",
        "date_range": "none",
        "person_name": "",
        "payment_app": "",
        "needs_payment_data": False,
        "understood_request": text
    }


# --------------------------------------------------
# PAYMENT CONTEXT
# --------------------------------------------------

def get_payment_context(session_id: str, interpretation: dict):
    session = get_session(session_id)
    today_payments = session["transactions"]

    date_range = interpretation.get("date_range", "none")

    if date_range == "today":
        selected = list(today_payments)
    elif date_range in ("yesterday", "previous"):
        selected = list(PREVIOUS_PAYMENTS)
    elif date_range == "all":
        selected = list(today_payments) + list(PREVIOUS_PAYMENTS)
    else:
        selected = list(today_payments) + list(PREVIOUS_PAYMENTS)

    person_name = (
        interpretation.get("person_name") or ""
    ).strip().lower()

    payment_app = (
        interpretation.get("payment_app") or ""
    ).strip().lower()

    if person_name:
        selected = [
            item for item in selected
            if person_name in item.get("name", "").lower()
        ]

    if payment_app:
        selected = [
            item for item in selected
            if payment_app in item.get("app", "").lower()
        ]

    return {
        "today": today_payments,
        "previous_day": PREVIOUS_PAYMENTS,
        "matching_records": selected
    }


# --------------------------------------------------
# BASIC ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "app": "Qunine",
        "status": "online",
        "version": "demo"
    }


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "app": "Qunine"
    }


@app.get("/api/welcome")
async def welcome():
    return {
        "message": (
            "Welcome to Qunine. I am here to help you "
            "understand your payments and make everyday tasks easier."
        )
    }


# --------------------------------------------------
# ONE-TIME STARTUP DEMO NOTIFICATION
# --------------------------------------------------

@app.get("/api/demo/notifications")
async def get_demo_notification(session_id: str):
    session = get_session(session_id)

    async with SESSION_LOCKS[session_id]:
        if session["notification_sent"]:
            return {
                "notification": None,
                "message": "The startup demonstration notification was already sent."
            }

        sample = random.choice(DEMO_NOTIFICATION_OPTIONS)
        now = datetime.now()

        notification = {
            "id": "demo_" + str(uuid.uuid4()),
            "name": sample["name"],
            "amount": sample["amount"],
            "app": sample["app"],
            "time": now.strftime("%I:%M %p"),
            "date": now.strftime("%d %b %Y"),
            "type": "simulated_demo",
            "status": "SIMULATED DEMO",
            "message": f"Simulated payment received from {sample['name']}"
        }

        session["notification_sent"] = True
        session["transactions"].append(notification)

        return {
            "notification": notification
        }


# --------------------------------------------------
# TRANSACTIONS
# --------------------------------------------------

@app.get("/api/transactions")
async def transactions(session_id: str):
    session = get_session(session_id)

    all_transactions = (
        list(session["transactions"]) + list(PREVIOUS_PAYMENTS)
    )

    return {
        "transactions": all_transactions,
        "total": len(all_transactions),
        "demo": True
    }


@app.get("/api/payments")
async def payments(session_id: str = "default"):
    session = get_session(session_id)

    return {
        "today": session["transactions"],
        "previous_day": PREVIOUS_PAYMENTS,
        "demo": True
    }


# --------------------------------------------------
# CHAT REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


# --------------------------------------------------
# CHAT ROUTING
# --------------------------------------------------

@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    session_id = request.session_id or "default"
    get_session(session_id)

    # GROQ 1: UNDERSTANDS THE USER'S MESSAGE

    understanding_result = await groq_request(
        GROQ_API_KEY_1,
        [
            {
                "role": "system",
                "content": UNDERSTANDING_PROMPT
            },
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=0.1
    )

    interpretation = extract_json(understanding_result)

    # GROQ 3: HANDLES STATEMENT QUESTIONS

    if interpretation.get("route") == "statement":
        statement_text = STATEMENTS.get(session_id)

        if not statement_text:
            answer = await groq_request(
                GROQ_API_KEY_3,
                [
                    {
                        "role": "system",
                        "content": STATEMENT_PROMPT
                    },
                    {
                        "role": "user",
                        "content": (
                            "The user has not uploaded a statement yet. "
                            "Explain that they can upload a PDF statement "
                            "to ask questions about it.\n\n"
                            f"User message: {message}"
                        )
                    }
                ]
            )

            return {
                "reply": answer,
                "route": "statement",
                "interpretation": interpretation
            }

        history = STATEMENT_HISTORY.setdefault(session_id, [])

        statement_messages = [
            {
                "role": "system",
                "content": STATEMENT_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Uploaded statement text:\n\n"
                    + statement_text[:50000]
                )
            }
        ]

        statement_messages.extend(history[-8:])

        statement_messages.append({
            "role": "user",
            "content": message
        })

        answer = await groq_request(
            GROQ_API_KEY_3,
            statement_messages,
            temperature=0.2
        )

        history.extend([
            {
                "role": "user",
                "content": message
            },
            {
                "role": "assistant",
                "content": answer
            }
        ])

        return {
            "reply": answer,
            "route": "statement",
            "interpretation": interpretation
        }

    # PYTHON: PREPARES PAYMENT INFORMATION

    payment_data = {}

    if interpretation.get("needs_payment_data"):
        payment_data = get_payment_context(
            session_id,
            interpretation
        )

    response_context = {
        "user_message": message,
        "understanding": interpretation,
        "payment_information": payment_data
    }

    # GROQ 2: GENERATES THE FINAL NORMAL RESPONSE

    answer = await groq_request(
        GROQ_API_KEY_2,
        [
            {
                "role": "system",
                "content": RESPONSE_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    response_context,
                    ensure_ascii=False
                )
            }
        ],
        temperature=0.5
    )

    return {
        "reply": answer,
        "route": "normal",
        "interpretation": interpretation
    }


# --------------------------------------------------
# PDF BANK STATEMENT ANALYSER
# --------------------------------------------------

@app.post("/api/analyze-statement")
async def analyze_statement(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="The PDF must be smaller than 10 MB."
        )

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        extracted_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF file."
        )

    if not extracted_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No selectable text was found in this PDF. "
                "Scanned image-only PDFs are not supported yet."
            )
        )

    extracted_text = extracted_text[:50000]

    STATEMENTS[session_id] = extracted_text
    STATEMENT_HISTORY[session_id] = []

    summary = await groq_request(
        GROQ_API_KEY_3,
        [
            {
                "role": "system",
                "content": STATEMENT_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Analyse this uploaded bank statement. "
                    "Give a short and clear summary. "
                    "Mention total credits and debits only if "
                    "the statement provides enough information.\n\n"
                    + extracted_text
                )
            }
        ],
        temperature=0.1
    )

    STATEMENT_HISTORY[session_id].append({
        "role": "assistant",
        "content": summary
    })

    return {
        "success": True,
        "reply": summary,
        "filename": file.filename,
        "message": "Statement processed for this session."
    }
