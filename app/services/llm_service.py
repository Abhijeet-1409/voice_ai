import json
import re

import google.generativeai as genai

from config.settings import settings
from services.rag_service import retrieve
from utils.logger import llm_logger

# ── Configure Gemini ──────────────────────────────────────────────────────────

genai.configure(api_key=settings.gemini_api_key)

_model = genai.GenerativeModel(settings.gemini_model)

llm_logger.info("Gemini model ready.")


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are Priya, a friendly and professional sales agent for Intelics Cloud Solutions.
You handle inbound sales calls from customers interested in cloud services.

YOUR PERSONA:
- Warm, helpful, and knowledgeable
- Speak naturally as if on a phone call — short sentences, conversational tone
- Never sound robotic or overly formal
- Use the customer's name if you know it

PRICING RULES (very important):
- ONLY quote prices from the pricing context provided to you
- NEVER quote prices from your own memory or training data
- If the pricing context is empty or does not cover what the customer asked — ask ONE clarifying question to narrow it down
- If the pricing context contains information but does not directly answer the customer's question — say so honestly. Do NOT make up prices, do NOT assume, do NOT fill gaps with guesses. Say something like "I have some information on that but let me get you the exact details" and ask a clarifying question
- After quoting a price — offer related services (e.g. after VM price, mention backup or storage options)

CONVERSATION RULES:
- Ask ONE question at a time — never bombard the customer with multiple questions
- Keep responses concise — this is a voice call, not an email
- If customer gives their name, use it naturally in the conversation
- If customer seems interested — offer to connect them with the sales team

INFORMATION COLLECTION RULES (critical — must follow every call):
- Your goal is to collect the customer's name, phone number and email before the call ends
- This is mandatory — every call must attempt to collect all three details
- Do NOT ask for all details at once — collect them one at a time, woven naturally into the conversation
- Ask for their name early — within the first 2 exchanges if they haven't given it
- Ask for their phone number after you have answered their main question and they seem satisfied
- Ask for their email last — after phone number is collected or if they decline to give phone
- If the customer seems hesitant — reassure them naturally: "Just so our team can follow up with you"
- If the customer declines to give any detail — acknowledge politely and move on, but try again naturally later in the conversation
- Never make the customer feel interrogated — keep it conversational and warm
- Do not end the call without attempting to collect all three details

RESPONSE FORMAT:
Always respond with a JSON object in this exact format:
{
    "reply": "your spoken response here"
}

Return ONLY the JSON object. No preamble. No markdown. No explanation outside the JSON.
""".strip()


EXTRACTION_PROMPT = """
You are an information extractor. Given a conversation exchange, extract any customer information mentioned.

Extract the following fields:
- caller_name: their first name or full name if mentioned, otherwise null
- caller_phone: phone number if mentioned, otherwise null
- caller_email: email address if mentioned, otherwise null
- caller_need: what they are looking for (brief summary), otherwise null
- interest_level: high / medium / low based on how engaged they seem, otherwise null

Respond ONLY with a JSON object in this exact format:
{
    "caller_name"    : "name or null",
    "caller_phone"   : "phone or null",
    "caller_email"   : "email or null",
    "caller_need"    : "brief summary or null",
    "interest_level" : "high or medium or low or null"
}

Return ONLY the JSON object. No preamble. No markdown.
""".strip()

# ── Stream reply ──────────────────────────────────────────────────────────────

async def stream_reply(
    transcript     : str,
    history        : list,
    session_id     : str,
):
    """
    Stream Gemini reply sentence by sentence.
    Calls RAG internally before Gemini to get pricing context.

    Args:
        transcript: Current customer message
        history:    List of previous exchanges from Redis session
        session_id: Session ID for log tracing

    Yields:
        One sentence at a time as a string.
    """

    # Stage 1 — get pricing context from RAG
    llm_logger.debug(f"[{session_id}] Fetching RAG context...")
    pricing_context = retrieve(transcript, session_id, history)

    # Stage 2 — build conversation history for Gemini
    gemini_history = _build_history(history)

    # Stage 3 — build current user message with pricing context
    user_message = _build_user_message(transcript, pricing_context)

    llm_logger.info(f"[{session_id}] Sending to Gemini (history={len(history)} exchanges)")

    # Stage 4 — stream from Gemini
    buffer = ""

    try:
        response = await _model.generate_content_async(
            [
                {"role": "user",  "parts": [SYSTEM_PROMPT]},
                {"role": "model", "parts": ["Understood. I am ready to help customers."]},
                *gemini_history,
                {"role": "user",  "parts": [user_message]},
            ],
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                buffer += chunk.text

        # Stage 5 — parse JSON from full response
        reply_text, _ = _parse_response(buffer, session_id)

        # Stage 6 — yield reply sentence by sentence for TTS
        sentences = _split_sentences(reply_text)
        llm_logger.info(f"[{session_id}] Reply split into {len(sentences)} sentences")

        for sentence in sentences:
            if sentence.strip():
                yield sentence.strip()

    except Exception as e:
        llm_logger.exception(f"[{session_id}] Gemini error: {e}")
        yield "I'm sorry, I had a small technical issue. Could you repeat that?"

async def extract_info(
    transcript  : str,
    agent_reply : str,
    session_id  : str,
) -> dict:
    """
    Extract customer information from one exchange.
    Called after each exchange completes — separate from the reply generation.

    Args:
        transcript:  Customer's message
        agent_reply: Agent's full reply text
        session_id:  Session ID for log tracing

    Returns:
        Dict with extracted fields — caller_name, caller_phone,
        caller_email, caller_need, interest_level.
        All values are strings or None.
    """
    exchange_text = f"Customer: {transcript}\nAgent: {agent_reply}"

    try:
        response = await _model.generate_content_async(
            [
                {"role": "user",  "parts": [EXTRACTION_PROMPT]},
                {"role": "model", "parts": ["Understood. I will extract customer information only."]},
                {"role": "user",  "parts": [exchange_text]},
            ],
            stream=False,
        )

        raw  = response.text
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)

        info = {
            "caller_name"    : data.get("caller_name"),
            "caller_phone"   : data.get("caller_phone"),
            "caller_email"   : data.get("caller_email"),
            "caller_need"    : data.get("caller_need"),
            "interest_level" : data.get("interest_level"),
        }

        llm_logger.debug(f"[{session_id}] Extraction result — {info}")
        return info

    except Exception as e:
        llm_logger.error(f"[{session_id}] extract_info failed — {e}")
        return {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_history(history: list) -> list:
    """Convert Redis exchanges list to Gemini message format."""
    messages = []
    for exchange in history:
        messages.append({
            "role"  : "user",
            "parts" : [exchange.get("caller_message", "")],
        })
        messages.append({
            "role"  : "model",
            "parts" : [exchange.get("agent_reply", "")],
        })
    return messages


def _build_user_message(transcript: str, pricing_context: str) -> str:
    """Build the user message — customer transcript + pricing context if available."""
    if pricing_context:
        return f"{transcript}\n\n[PRICING CONTEXT]\n{pricing_context}"
    return transcript


def _parse_response(raw: str, session_id: str) -> tuple[str, dict]:
    """
    Parse Gemini JSON response into reply text.
    Falls back gracefully if JSON is malformed.
    """
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)
        reply = data.get("reply", "").strip()
        llm_logger.debug(f"[{session_id}] JSON parsed successfully")
        return reply, {}

    except (json.JSONDecodeError, AttributeError) as e:
        llm_logger.warning(f"[{session_id}] JSON parse failed: {e} | raw: {raw[:200]}")
        return raw.strip(), {}


def _split_sentences(text: str) -> list[str]:
    """
    Split reply text into sentences for sentence-by-sentence TTS.
    Splits on . ? ! followed by space or end of string.
    """
    sentences = re.split(r'(?<=[.?!])(?:[\"\']?\s+|\s*$)', text)
    return [s.strip() for s in sentences if s.strip()]