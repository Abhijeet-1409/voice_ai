import json
import re

import google.generativeai as genai

from config.settings import settings
from services.rag_service import retrieve

# ── Configure Gemini ──────────────────────────────────────────────────────────

genai.configure(api_key=settings.gemini_api_key)

_model = genai.GenerativeModel("gemini-2.5-flash")

print("[llm_service] Gemini model ready.")


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
- After quoting a price — offer related services (e.g. after VM price, mention backup or storage options)

CONVERSATION RULES:
- Ask ONE question at a time — never bombard the customer with multiple questions
- Keep responses concise — this is a voice call, not an email
- If customer gives their name, use it naturally in the conversation
- If customer seems interested — offer to connect them with the sales team

EXTRACTION RULES:
At the end of your JSON response, always extract any information mentioned by the customer:
- caller_name: their first name or full name if mentioned
- caller_phone: phone number if mentioned
- caller_email: email address if mentioned
- caller_need: what they are looking for (brief summary)
- interest_level: high / medium / low based on how engaged they seem

RESPONSE FORMAT:
Always respond with a JSON object in this exact format:
{
    "reply"          : "your spoken response here",
    "caller_name"    : "name or null",
    "caller_phone"   : "phone or null",
    "caller_email"   : "email or null",
    "caller_need"    : "brief summary or null",
    "interest_level" : "high or medium or low or null"
}

Return ONLY the JSON object. No preamble. No markdown. No explanation outside the JSON.
""".strip()


# ── Stream reply ──────────────────────────────────────────────────────────────

async def stream_reply(
    transcript     : str,
    history        : list,
    extracted_info : dict | None = None,
):
    """
    Stream Gemini reply sentence by sentence.
    Calls RAG internally before Gemini to get pricing context.

    Args:
        transcript:     Current customer message
        history:        List of previous exchanges from Redis session
        extracted_info: Mutable dict — updated with any newly extracted info

    Yields:
        One sentence at a time as a string.
        Also updates extracted_info dict in place if provided.
    """

    # Stage 1 — get pricing context from RAG
    pricing_context = retrieve(transcript, history)

    # Stage 2 — build conversation history for Gemini
    gemini_history = _build_history(history)

    # Stage 3 — build current user message with pricing context
    user_message = _build_user_message(transcript, pricing_context)

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
        reply_text, info = _parse_response(buffer)

        # Update extracted info in place
        if extracted_info is not None and info:
            for field in ("caller_name", "caller_phone", "caller_email",
                          "caller_need", "interest_level"):
                value = info.get(field)
                if value and extracted_info.get(field) is None:
                    extracted_info[field] = value

        # Stage 6 — yield reply sentence by sentence for TTS
        sentences = _split_sentences(reply_text)
        for sentence in sentences:
            if sentence.strip():
                yield sentence.strip()

    except Exception as e:
        print(f"[llm_service] Gemini error: {e}")
        yield "I'm sorry, I had a small technical issue. Could you repeat that?"


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


def _parse_response(raw: str) -> tuple[str, dict]:
    """
    Parse Gemini JSON response into reply text and extracted info dict.
    Falls back gracefully if JSON is malformed.
    """
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)

        reply = data.get("reply", "").strip()
        info  = {
            "caller_name"    : data.get("caller_name"),
            "caller_phone"   : data.get("caller_phone"),
            "caller_email"   : data.get("caller_email"),
            "caller_need"    : data.get("caller_need"),
            "interest_level" : data.get("interest_level"),
        }
        return reply, info

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[llm_service] JSON parse failed: {e} | raw: {raw[:200]}")
        # Return raw text as reply if JSON parsing fails
        return raw.strip(), {}


def _split_sentences(text: str) -> list[str]:
    """
    Split reply text into sentences for sentence-by-sentence TTS.
    Splits on . ? ! followed by space or end of string.
    """
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if s.strip()]