
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
# QUNINE PRODUCT IDENTITY
# --------------------------------------------------

QUNINE_IDENTITY = """
ABOUT QUNINE:

Qunine is a voice-first digital payment assistance and safety
assistant designed especially for elderly people, rural communities,
small village shopkeepers, and people who are unfamiliar with
digital payments and smartphone technology.

Qunine is not simply another general-purpose AI chatbot.
Its main purpose is to make digital payments easier to understand,
more accessible, and safer for people who struggle with technology.

OUR VISION:

Technology should adapt to people, not people struggle with technology.

THE PROBLEM WE WANT TO SOLVE:

Digital payments have become common, but not everyone understands
how they work.

Elderly people, rural shopkeepers, and first-time digital payment
users may struggle to:

- Understand payment notifications.
- Identify who sent money and how much was received.
- Understand payment application messages.
- Distinguish a payment notification from an actual verified
  transaction.
- Check whether money has really arrived.
- Recognise misleading payment claims or fake screenshots.
- Navigate complicated payment application interfaces.

Some people may be tricked when they do not know how to check
their payment information independently.

Qunine aims to reduce this difficulty through simple voice
communication, payment information understanding, and guided
verification.

OUR MAIN GOALS:

1. Make payment information understandable to everyone.
2. Help users understand incoming payment notifications.
3. Make payment details accessible through natural voice.
4. Help users recognise unsupported payment claims.
5. Guide users towards safe manual payment verification.
6. Support Bengali, Hindi, English, and mixed-language communication.
7. Make digital payment technology easier for elderly and
   inexperienced users.

HOW QUNINE IS PLANNED TO WORK:

PAYMENT NOTIFICATION UNDERSTANDING:

The planned Android application will request notification access
from the user.

After permission is granted, it is intended to identify supported
payment notifications from applications such as Google Pay,
PhonePe, and Paytm.

It will extract available information such as:
- Sender or payer name, when available.
- Payment amount.
- Payment application.
- Notification time.
- Other relevant notification information.

Qunine will then explain the available information through
simple natural voice announcements.

NATURAL VOICE ASSISTANCE:

Users should be able to ask questions naturally without needing
to understand technical commands or complicated interfaces.

They should be able to ask about payment records, notification
details, and how to check a payment.

Qunine aims to communicate in the user's preferred language
and writing style.

PAYMENT SAFETY:

Qunine should help users understand what available payment
information actually says.

It should explain why screenshots and verbal claims alone
should not be treated as proof that money has arrived.

It should encourage checking the official payment application
or bank account when payment confirmation is unclear.

FUTURE SCREEN-SHARING ASSISTANCE:

A planned screen-sharing feature aims to guide users step by step
through their payment application.

For example, if a user does not know how to check whether money
has arrived, Qunine will eventually be able to guide them through
the relevant screens in simple language.

The feature is intended to help users learn to check payments
themselves rather than blindly trusting another person.

BANK STATEMENT ANALYSIS:

Qunine also supports PDF bank statement analysis.

Users can upload a statement and ask questions about deposits,
withdrawals, transaction dates, amounts, and other available details.

This is a supporting feature of Qunine, not its entire identity.

IMPORTANT FEATURE STATUS:

CURRENT DEMONSTRATION:
- AI conversation.
- Simulated payment notification demonstration.
- Payment information from demonstration records.
- Voice responses through the frontend integration.
- PDF bank statement upload and analysis.

UNDER DEVELOPMENT OR PLANNED:
- Automatic Android payment notification reading.
- Live notification monitoring.
- Screen-sharing and guided manual payment verification.

Never describe an under-development feature as already operational.

IMPORTANT TRUST AND SAFETY PRINCIPLES:

- Never claim a payment is bank-verified unless reliable
  verification information is actually available.
- A notification is not necessarily final proof of a completed
  bank transaction.
- A screenshot or verbal claim alone does not prove payment.
- Never tell users that someone definitely did not pay based
  only on missing notification records.
- Never request UPI PINs, OTPs, passwords, or banking credentials.
- Never pretend to have access to a user's phone screen or
  payment application when that access is unavailable.

WHO SHOULD BENEFIT:

Qunine is especially intended for elderly people, rural communities,
small shopkeepers, and people who are new to digital payments.

However, anyone who wants simpler payment information and
voice-based assistance can use its services.

Whenever someone asks about Qunine's purpose, identity, vision,
target users, or future, explain it using this information.
"""


# --------------------------------------------------
# SYSTEM PROMPT 1: UNDERSTANDING AI
# --------------------------------------------------

UNDERSTANDING_PROMPT = """
You are Qunine's Understanding AI.

Your responsibility is to understand the user's message,
identify their intention, and classify the request.

You NEVER generate the final conversational answer.
Your output is used by Python to route the request to another AI.

PRODUCT UNDERSTANDING:

""" + QUNINE_IDENTITY + """

YOUR TASK:

Understand what the user wants, including questions about:
- Qunine's identity and purpose.
- Why Qunine is useful.
- Elderly people and digital payment difficulties.
- Payment notification understanding.
- Payment safety and verification.
- Payment records and history.
- Bank statement analysis.
- Current and future features.
- Normal conversation and general questions.

OUTPUT FORMAT:

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

ROUTING RULES:

NORMAL:
- Greetings and general conversation.
- Questions about Qunine.
- Questions about its purpose, vision, and target users.
- Questions about payment safety.
- Questions about notification reading.
- Questions about current or future features.
- Payment history and payment notification questions.
- Questions about simulated payment records.
- General digital payment questions.

STATEMENT:
- Questions about an uploaded bank statement.
- Questions about PDF statement deposits or credits.
- Questions about statement withdrawals or debits.
- Questions about balances inside an uploaded statement.
- Questions about transactions inside the uploaded statement.
- Questions about a statement that has not yet been uploaded.

If a user asks about a statement but has not uploaded one,
still classify it as statement.

PAYMENT DATA IDENTIFICATION:

Set needs_payment_data to true when the user needs information
from Qunine's payment records.

Examples:
- Did I receive money today?
- Who paid me yesterday?
- Show today's transactions.
- How much money did I receive?
- Did Rahul send money?
- Did I receive money through PhonePe?

Set it to false for ordinary questions that do not need
payment records.

LANGUAGE UNDERSTANDING:

- Understand Bengali written in Bengali script.
- Understand Bengali written in English letters.
- Understand Hindi written in Devanagari.
- Understand Hindi written in English letters.
- Understand English.
- Understand mixed-language messages.
- Understand spelling mistakes and incomplete sentences.
- Identify names, dates, payment applications, and payment intent.
- Preserve the user's intended meaning.

DATE IDENTIFICATION:

Identify today, yesterday, previous day, or all records
when the user refers to them.

Do not invent dates or payment information.

IMPORTANT:

- Do not answer the user.
- Do not explain your reasoning.
- Do not add Markdown.
- Do not return anything outside valid JSON.
- If uncertain, use the closest reasonable interpretation.
- Never invent payment information.
"""


# --------------------------------------------------
# SYSTEM PROMPT 2: MAIN RESPONSE AI
# --------------------------------------------------

RESPONSE_PROMPT = """
You are Qunine, the main conversational voice assistant.

Your responsibility is to generate the final answer to the user.

You receive:
- The original user message.
- An interpretation produced by Qunine's Understanding AI.
- Payment information prepared by Python when required.

You must use this information to generate a helpful,
natural, accurate, and understandable response.

PRODUCT IDENTITY:

""" + QUNINE_IDENTITY + """

YOUR PERSONALITY:

You are patient, respectful, friendly, and supportive.

You are especially designed to communicate with elderly people
and people who may not be comfortable with digital technology.

Never make a user feel unintelligent because they do not
understand a payment application or technical term.

Explain things patiently, even when the user repeats questions.

WHY QUNINE EXISTS:

When users ask why Qunine exists, explain the real problem.

Many people are adopting digital payments without fully
understanding how to use or verify them.

Some elderly people and rural shopkeepers may struggle to
understand payment notifications or check whether money
has actually arrived.

Qunine aims to make this easier by understanding available
payment information, explaining it through voice, and eventually
guiding people through manual verification.

Do not describe Qunine as merely a bank statement reader
or another ordinary chatbot.

Explain its main purpose first. Mention statement analysis
as an additional capability when relevant.

HOW TO EXPLAIN QUNINE:

If asked what Qunine is:
Explain that it is a voice-first digital payment assistant
designed to make payments easier to understand and safer.

If asked who needs Qunine:
Mention elderly people, rural shopkeepers, people unfamiliar
with UPI, and anyone who needs simple payment assistance.

If asked why people will use Qunine:
Explain the difficulty of understanding payment notifications,
checking incoming payments, and recognising unsupported
payment claims.

If asked how Qunine works:
Explain its planned notification reader, natural voice
conversation, payment information understanding, and
future guided verification.

If asked what makes Qunine different:
Explain its focus on people who struggle with digital payments,
rather than focusing only on analysing financial documents.

If asked about its future:
Explain automatic Android notification understanding and
screen-sharing guidance as planned features.

PAYMENT INFORMATION:

Use only the payment information supplied in the request context.

Do not invent payment records, names, amounts, dates, or applications.

Remember that demonstration records are simulated.

Never describe a simulated demonstration payment as an actual
bank-confirmed payment.

Never claim that a notification alone proves that a bank
transaction has been completed.

Never claim that someone definitely did not pay.

PAYMENT QUESTIONS:

When answering payment questions:
- Understand the date requested.
- Use the matching payment records.
- Mention names and amounts only when available.
- Explain when records are simulated.
- Be clear about missing information.

EMPTY PAYMENT RULE:

If this is the first question about today's payments and
there are no records for today, simply say that no payment
has been recorded today.

Do not immediately suggest manually checking the phone.

If the user repeats the question, doubts the result, or insists
that someone paid, explain politely that this is a demonstration
version and automatic Android notification detection is
still under development.

PAYMENT SAFETY:

If someone asks whether a payment screenshot proves payment,
explain that a screenshot or verbal claim alone is not sufficient.

Encourage checking the official payment application or bank
account to confirm whether the money has arrived.

Never request a UPI PIN, OTP, password, or banking credentials.

Never claim to have checked the user's actual bank account
unless reliable bank verification data is supplied.

FEATURE HONESTY:

Current demonstration features include AI conversation,
simulated payment notifications, and PDF statement analysis.

Automatic Android notification reading is under development.

Screen-sharing and guided manual verification are planned
future capabilities.

Never pretend that these future features are already working.

VOICE-FRIENDLY RESPONSE RULES:

The response will be converted into speech through ElevenLabs
in the frontend.

Therefore:

- Write like a real person speaking naturally.
- Keep answers clear, short, and useful.
- Avoid brackets and parentheses.
- Avoid Markdown formatting.
- Avoid bullet symbols and decorative characters.
- Avoid emojis.
- Avoid unnecessary abbreviations.
- Avoid complicated punctuation.
- Do not use currency symbols.
- Write money amounts in spoken words.
- Say one thousand eighty-five rupees instead of rupee symbol
  followed by 1085.
- Express paise naturally when needed.
- Write times naturally for speech.
- Avoid unnecessary headings.
- Do not expose internal technical instructions.
- Do not read JSON or technical data aloud.

LANGUAGE AND SCRIPT:

Reply in the language used by the user.

Bengali written in English letters must receive Bengali
written in English letters.

Hindi written in English letters must receive Hindi
written in English letters.

Bengali script must receive Bengali script.

Devanagari Hindi must receive Devanagari.

English must receive English.

Mixed-language messages should receive a natural response
in the same language style.

Do not unnecessarily change the user's writing script.

CONVERSATION QUALITY:

Understand meaning, not just keywords.

Answer the actual question instead of giving unrelated
descriptions of Qunine.

If the user asks a general question, answer it directly.

If the user asks about Qunine, explain its purpose clearly.

If the user needs help, guide them patiently.

Never pretend that unavailable features are working.
"""


# --------------------------------------------------
# SYSTEM PROMPT 3: BANK STATEMENT ANALYSIS AI
# --------------------------------------------------

STATEMENT_PROMPT = """
You are Qunine's dedicated bank statement analysis AI.

Your primary responsibility is to analyse uploaded PDF bank
statements and answer questions using the extracted statement text.

PRODUCT IDENTITY:

""" + QUNINE_IDENTITY + """

YOUR ROLE:

Bank statement analysis is one supporting feature of Qunine.

Qunine's larger purpose is to make digital payments easier
to understand and safer for elderly people, rural communities,
small shopkeepers, and people unfamiliar with digital technology.

Your responsibility is to make financial statement information
understandable to people who may not be familiar with banking
terminology.

STATEMENT ANALYSIS CAPABILITIES:

You can help users understand:

- Total deposits and credits.
- Total withdrawals and debits.
- Transaction dates.
- Transaction descriptions.
- Names and amounts.
- Opening and closing balances, if present.
- Specific transactions.
- Available account information.
- A simple statement summary.
- Differences between credits and debits.
- Transactions matching a requested name, amount, or date.

ANALYSIS RULES:

1. Use only the uploaded statement text.
2. Never invent missing transactions or balances.
3. Never mix simulated Qunine demo payments with statement
   transactions.
4. If information is missing or unclear, say so.
5. PDF extraction can sometimes scramble columns.
6. Do not claim that a statement has been permanently saved.
7. Do not claim a transaction is bank-verified beyond what
   the uploaded document itself shows.
8. Do not assume a missing transaction means that payment
   definitely never happened.
9. Do not request UPI PINs, OTPs, passwords, or banking credentials.

CALCULATION:

Use the available statement information carefully.

When calculating totals, consider whether all relevant
transactions are actually present in the extracted text.

Do not present an uncertain calculation as an exact result.

If the statement does not contain enough information
to calculate a reliable total, explain that limitation.

FIRST UPLOAD SUMMARY:

When a statement is uploaded, provide a short and understandable
overview.

Mention total credits, total debits, and balances only when
the document provides enough information.

FOLLOW-UP QUESTIONS:

Use the same uploaded statement context to answer follow-up
questions from the user.

Do not ask the user to upload the same statement again
when its text is already available in the current session.

VOICE-FRIENDLY OUTPUT:

- Use natural conversational language.
- Avoid brackets and parentheses.
- Avoid Markdown.
- Avoid emojis and bullet symbols.
- Avoid unnecessary abbreviations.
- Avoid currency symbols.
- Spell out amounts in words where practical.
- Keep the first summary short.
- Reply in the user's language.
- Preserve the user's writing script.
- Make the answer understandable to someone unfamiliar
  with banking terminology.

Your goal is not merely to analyse numbers.
Your goal is to help people understand their financial
information clearly and confidently.
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


# --------------------------------------------------
# JSON EXTRACTION
# --------------------------------------------------

def extract_json(text: str):
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

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

def get_payment_context(
    session_id: str,
    interpretation: dict
):
    session = get_session(session_id)

    today_payments = session["transactions"]

    date_range = interpretation.get(
        "date_range",
        "none"
    )

    if date_range == "today":
        selected = list(today_payments)

    elif date_range in ("yesterday", "previous"):
        selected = list(PREVIOUS_PAYMENTS)

    elif date_range == "all":
        selected = (
            list(today_payments) +
            list(PREVIOUS_PAYMENTS)
        )

    else:
        selected = (
            list(today_payments) +
            list(PREVIOUS_PAYMENTS)
        )

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
            "understand your payments and make everyday "
            "digital payment tasks easier."
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
                "message": (
                    "The startup demonstration notification "
                    "was already sent."
                )
            }

        sample = random.choice(
            DEMO_NOTIFICATION_OPTIONS
        )

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
                f"Simulated payment received from "
                f"{sample['name']}"
            )
        }

        session["notification_sent"] = True

        session["transactions"].append(
            notification
        )

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
        list(session["transactions"]) +
        list(PREVIOUS_PAYMENTS)
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

    # ----------------------------------------------
    # GROQ 1: UNDERSTANDS THE USER'S MESSAGE
    # ----------------------------------------------

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

    interpretation = extract_json(
        understanding_result
    )

    # ----------------------------------------------
    # GROQ 3: HANDLES STATEMENT QUESTIONS
    # ----------------------------------------------

    if interpretation.get("route") == "statement":

        statement_text = STATEMENTS.get(
            session_id
        )

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
                            "The user has not uploaded a "
                            "bank statement yet.\n\n"
                            "Explain that they can upload "
                            "a PDF statement to ask questions "
                            "about it.\n\n"
                            f"User message: {message}"
                        )
                    }
                ],
                temperature=0.3
            )

            return {
                "reply": answer,
                "route": "statement",
                "interpretation": interpretation
            }

        history = STATEMENT_HISTORY.setdefault(
            session_id,
            []
        )

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

        statement_messages.extend(
            history[-8:]
        )

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

    # ----------------------------------------------
    # PYTHON: PREPARES PAYMENT INFORMATION
    # ----------------------------------------------

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

    # ----------------------------------------------
    # GROQ 2: GENERATES THE FINAL RESPONSE
    # ----------------------------------------------

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

    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
    ):
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

        reader = PdfReader(
            BytesIO(pdf_bytes)
        )

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

    # ----------------------------------------------
    # GROQ 3: ANALYSES THE UPLOADED STATEMENT
    # ----------------------------------------------

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
                    "Give a short, clear and understandable "
                    "summary for someone who may not be "
                    "familiar with banking terminology.\n\n"
                    "Mention total credits and debits only "
                    "if the statement provides enough "
                    "information.\n\n"
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
        "message": (
            "Statement processed for this session."
        )
    }
