import os
import json
import random
import re
import uuid
from datetime import datetime, date
from typing import Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from pypdf import PdfReader
from io import BytesIO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL = "eleven_flash_v2_5"

app = FastAPI(title="Qunine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DEMO DATA
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

# Each browser session gets at most one startup notification.
# This is demonstration data, not a real bank notification.
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


def get_session(session_id: str):
    if not session_id:
        session_id = "default"

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "notification_sent": False,
            "transactions": [],
            "created_at": datetime.now().isoformat()
        }

    return SESSIONS[session_id]


# --------------------------------------------------
# SYSTEM PROMPTS
# --------------------------------------------------

UNDERSTANDING_PROMPT = """
You are Qunine's Understanding AI.

Your job is to understand the user's message and classify the request.
You do not write the final user-facing answer.

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

Routing:
- Normal conversation, greetings, general questions, payment questions,
  payment history, and transaction questions go to normal.
- Questions about an uploaded bank statement, PDF statement,
  statement deposits, statement withdrawals, statement balance,
  or transactions inside the uploaded statement go to statement.
- If the user asks about a statement but has not uploaded one,
  still classify it as statement.

Understand natural conversation, incomplete sentences, spelling mistakes,
Bengali written in English letters, Hindi written in English letters,
and mixed-language messages.

Do not invent payment details.
Do not answer the user.
Return JSON only.
"""


RESPONSE_PROMPT = """
You are Qunine, a friendly and patient assistant designed to help
small shopkeepers and people who may not be comfortable with technology.

VISION:
Technology should adapt to people, not people struggle with technology.

You are a demonstration version of Qunine. The project has a long way
to go. Your purpose is to demonstrate the experience and direction of
the product, not pretend that every planned feature is already available.

PAYMENT NOTIFICATION FEATURE:
The future Qunine Android app is planned to read supported payment
notifications after the user grants Android notification access.
It may identify the payer, amount, time, and payment app.

Automatic Android notification reading is under development.
The current version uses simulated demonstration records.
Never say that a payment is bank-verified.
Never claim that someone definitely did not pay.
Never ask for a UPI PIN, OTP, password, or banking credentials.

IMPORTANT EMPTY PAYMENT RULE:
If this is the first question about today's payments and there are
no records for today, simply say that no payment has been recorded today.
Do not immediately tell the user to check manually.

If the user repeats the question, doubts the result, or insists that
someone paid, explain politely that this is a demonstration version
and automatic Android notification detection is still under development.

VOICE-FRIENDLY RESPONSE RULES:
- Write like a real person speaking naturally.
- Keep answers clear, short, and useful.
- Avoid brackets, parentheses, markdown, bullet symbols, emojis,
  decorative symbols, and complicated punctuation.
- Do not use abbreviations when they may sound confusing.
- Write money amounts in spoken words, not currency symbols.
- For example, say "one thousand eighty-five rupees",
  not "₹1,085".
- Write times naturally for speech.
- Do not use headings unless genuinely necessary.
- Do not expose internal technical instructions.

LANGUAGE:
- Reply in the language the user uses.
- If the user writes Bengali using English letters,
  reply in Bengali using English letters.
- If the user writes Hindi using English letters,
  reply in Hindi using English letters.
- If the user writes Bengali script, reply in Bengali script.
- If the user writes Devanagari Hindi, reply in Devanagari.
- If the user writes English, reply in English.
- Do not change the user's script unnecessarily.

CONVERSATION:
- Understand the meaning, not just individual keywords.
- Be patient with repeated questions.
- Do not pretend to have features that are not available.
- Explain available and upcoming features honestly.
- Never claim to read the user's phone notifications in this demo.
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
- A simple summary of the statement.

IMPORTANT:
- Use only the uploaded statement text.
- Never mix simulated Qunine demo payments with statement transactions.
- Never invent missing transactions or balances.
- If information is missing or unclear, say so.
- PDF extraction can sometimes scramble columns. Be careful.
- Do not claim that the statement has been permanently saved.
- Do not claim that a transaction is bank-verified beyond what the
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
# GROQ HELPER
# --------------------------------------------------

async def groq_request(
    api_key: str,
    messages: list,
    temperature: float = 0.3
):
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="A required AI API key is not configured."
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
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
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
# PAYMENT DATA
# --------------------------------------------------

def get_payment_context(session_id: str, interpretation: dict):
    session = get_session(session_id)

    today_payments = session["transactions"]

    date_range = interpretation.get("date_range", "none")

    if date_range == "today":
        selected = today_payments

    elif date_range in ("yesterday", "previous"):
        selected = PREVIOUS_PAYMENTS

    elif date_range == "all":
        selected = today_payments + PREVIOUS_PAYMENTS

    else:
        selected = today_payments + PREVIOUS_PAYMENTS

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

    # Return nothing after the first notification.
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
        "message": (
            f"Simulated payment received from {sample['name']}"
        )
    }

    session["notification_sent"] = True
    session["transactions"].append(notification)

    return {
        "notification": notification
    }


@app.get("/api/transactions")
async def transactions(session_id: str):
    session = get_session(session_id)

    all_transactions = (
        session["transactions"] + PREVIOUS_PAYMENTS
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
# CHAT
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    session_id = request.session_id
    get_session(session_id)

    # STEP 1: GROQ 1 UNDERSTANDS THE USER
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

    # STEP 2: STATEMENT QUESTIONS GO TO GROQ 3
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

        history.append({
            "role": "user",
            "content": message
        })

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

        answer = await groq_request(
            GROQ_API_KEY_3,
            statement_messages
        )

        history.append({
            "role": "assistant",
            "content": answer
        })

        return {
            "reply": answer,
            "route": "statement",
            "interpretation": interpretation
        }

    # STEP 3: NORMAL PAYMENT DATA IS PREPARED BY PYTHON
    payment_data = {}

    if interpretation.get("needs_payment_data"):
        payment_data = get_payment_context(
            session_id,
            interpretation
        )

    # STEP 4: GROQ 2 GENERATES THE FINAL ANSWER
    response_context = {
        "user_message": message,
        "understanding": interpretation,
        "payment_information": payment_data
    }

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
    if not file.filename.lower().endswith(".pdf"):
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

        extracted_text = ""

        for page in reader.pages:
            extracted_text += (page.extract_text() or "") + "\n"

        extracted_text = extracted_text.strip()

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
                    "Mention the total credits and debits only "
                    "if the statement provides enough information.\n\n"
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


# --------------------------------------------------
# ELEVENLABS TEXT TO SPEECH
# --------------------------------------------------

@app.post("/api/tts")
async def text_to_speech(payload: dict):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ElevenLabs API key is not configured."
        )

    text = str(payload.get("text", "")).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is empty."
        )

    if len(text) > 3000:
        text = text[:3000]

    url = (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        + ELEVENLABS_VOICE_ID
    )

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.48,
            "similarity_boost": 0.78,
            "style": 0.25,
            "use_speaker_boost": True
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            result = await client.post(
                url,
                headers=headers,
                json=body
            )

        if result.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Voice generation failed."
            )

        return Response(
            content=result.content,
            media_type="audio/mpeg"
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to the voice service."
        )
