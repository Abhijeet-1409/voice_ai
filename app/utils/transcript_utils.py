"""
transcript_utils.py — Transcript post-processing utilities

Two-stage correction pipeline:
  Stage 1: Regex fixes — zero latency, zero cost
           Handles common spoken patterns (double X, at symbol, dot extensions)
  Stage 2: Gemini correction — only when contact info detected
           Handles emails, phone numbers, complex spoken patterns

Used in websocket_handler._process_exchange() after STT finalize().

NOTE: _DOUBLE_MAP covers digit words only (zero-nine).
      Letter spelling like "double u" (→ W) is intentionally left to Gemini.

COVERAGE:
  Regex handles  — double digits, dot extensions, oh-as-zero, known email domains
  Gemini handles — fully spoken phone numbers, unknown domains, underscore/hyphen,
                   mixed digit+word numbers, character-by-character spelling
  Not handled    — Hindi/non-English digit words (out of scope)

BUGS FIXED vs first version:
  1. dot co dot in — Fix 0 runs BEFORE Fix 2 with \\s* to absorb spaces
  2. dot extensions leaving space artifacts — \\s* before each extension fix
  3. \\bdouble\\b replaced with \\bdouble\\s+\\d — only fires for digit doubles,
     not "double u" (letter W spelling)
  4. \\boh\\b added to _CONTACT_PATTERNS — catches oh-as-zero in phone numbers
  5. _has_contact_info called on ORIGINAL text BEFORE regex runs —
     prevents "looking at Linux" being corrupted and going undetected
  6. 'company' removed from _EMAIL_CONTEXT — too generic, false positives
"""

import re

import google.generativeai as genai

from config.settings import settings
from utils.logger import transcript_logger

# ── Gemini model for transcript correction ────────────────────────────────────
# Separate lightweight instance — no system prompt, no history, no RAG
# Created once at module load, reused for every correction call

genai.configure(api_key=settings.transcript_gemini_api_key)
_correction_model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# ── Digit words — used in both regex and contact detection ────────────────────

_DIGIT_WORDS = {
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'oh',
}

_DIGIT_WORD_PATTERN = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|oh)\b'

# ── Contact info detection patterns ──────────────────────────────────────────

_CONTACT_PATTERNS = [
    # Email signals
    r'\bgmail\b',
    r'\byahoo\b',
    r'\boutlook\b',
    r'\bhotmail\b',
    r'\brediffmail\b',
    r'\bprotonmail\b',
    r'\bunderscore\b',
    r'\bhyphen\b',
    r'dot com',
    r'dot in',
    r'dot net',
    r'dot org',
    r'dot io',
    r'dot co',

    # Phone number signals
    r'\b\d{5,}\b',              # 5+ digit sequence
    r'\bdouble\s+\d',           # FIX: "double" only when followed by digit
                                # was \bdouble\b — caused "double u" false positive
    r'\boh\b',                  # FIX: added — oh as spoken zero in phone numbers

    # Explicit mention signals
    r'my email',
    r'my phone',
    r'my number',
    r'my contact',
    r'call me on',
    r'reach me at',
    r'contact me at',
    r'you can reach me',
    r'you can call me',

    # Gap 2 fix — "at" present alongside a domain extension
    # handled dynamically in _has_contact_info()

    # Gap 1 fix — two or more spoken digit words = likely phone number
    # handled dynamically in _has_contact_info()
]


def _has_contact_info(text: str) -> bool:
    """
    Returns True if transcript likely contains email or phone number.

    MUST be called on original text BEFORE _regex_fix() runs.
    correct_transcript() enforces this ordering — if called after regex,
    patterns like \\bat\\b miss cases where regex already converted "at" to "@".

    Uses static patterns plus two dynamic checks:
      - Gap 1: two or more spoken digit words → likely phone number
      - Gap 2: "at" present alongside any domain extension → likely email
    """
    lower = text.lower()

    # Static pattern check
    for pattern in _CONTACT_PATTERNS:
        if re.search(pattern, lower):
            return True

    # Gap 1 fix — count spoken digit words (including "oh")
    # If 2 or more digit words appear → probably a phone number being dictated
    digit_word_matches = re.findall(_DIGIT_WORD_PATTERN, lower)
    if len(digit_word_matches) >= 2:
        transcript_logger.debug("Contact detection — 2+ digit words found — flagging for Gemini")
        return True

    # Gap 2 fix — "at" + any domain extension anywhere in text
    # Catches unknown company domains like "john at tatacloud dot com"
    has_at     = bool(re.search(r'\bat\b', lower))
    has_domain = bool(re.search(r'dot\s+(?:com|in|net|org|io|co|edu|gov)', lower))
    if has_at and has_domain:
        transcript_logger.debug("Contact detection — 'at' + domain extension found — flagging for Gemini")
        return True

    return False


# ── Regex pre-fixes ───────────────────────────────────────────────────────────

# Digit words only — "double u" (→ W) is intentionally left to Gemini
_DOUBLE_MAP = {
    'zero'  : '00', 'one'   : '11', 'two'   : '22', 'three' : '33',
    'four'  : '44', 'five'  : '55', 'six'   : '66', 'seven' : '77',
    'eight' : '88', 'nine'  : '99',
}

# Known email providers for context-aware "at" → "@" conversion
# For unknown domains, Gap 2 fix ensures Gemini handles them
# NOTE: 'company' removed — too generic, causes false positives
_EMAIL_CONTEXT = {
    'gmail', 'yahoo', 'outlook', 'hotmail', 'rediffmail',
    'protonmail', 'intelics', 'writer',
}


def _regex_fix(text: str) -> str:
    """
    Fast regex fixes for common spoken patterns.
    Applied after _has_contact_info() check on original — never before.

    Fixes applied in order (ORDER MATTERS):
      0. Compound domains (dot co dot in) → .co.in  ← MUST run before Fix 2
      1. "double <digit>" → repeated digit
      2. Domain extensions dot com/in/net/org/io/co → .com/.in etc
         Uses \\s* to absorb space artifacts left by Fix 0
      3. "oh" → "0" only between digit characters
      4. "word at known-domain" → "word@known-domain"
         Unknown domains left to Gemini via Gap 2 fix
    """

    # Fix 0 — compound Indian domain FIRST — must run before Fix 2
    # If Fix 2 runs first: "dot in" → ".in" → "dot co .in" → Fix 0 never matches
    # \\s* absorbs any space between the word and the compound suffix
    text = re.sub(
        r'\s*\bdot\s+co\s+dot\s+in\b',
        '.co.in',
        text,
        flags=re.IGNORECASE,
    )

    # Fix 1 — "double <digit word>" → repeated digit
    def replace_double(m):
        word = m.group(1).lower()
        return _DOUBLE_MAP.get(word, m.group(0))

    text = re.sub(
        r'\bdouble\s+(zero|one|two|three|four|five|six|seven|eight|nine)\b',
        replace_double,
        text,
        flags=re.IGNORECASE,
    )

    # Fix 2 — domain extensions
    # \\s* absorbs space artifacts left by Fix 0 and Fix 1
    # Without \\s*: "rahul@gmail .com" — space before extension
    text = re.sub(r'\s*\bdot\s+com\b',  '.com', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+in\b',   '.in',  text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+net\b',  '.net', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+org\b',  '.org', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+io\b',   '.io',  text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+co\b',   '.co',  text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+edu\b',  '.edu', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\bdot\s+gov\b',  '.gov', text, flags=re.IGNORECASE)

    # Fix 3 — "oh" → "0" only when sandwiched between digit characters
    # "9 oh 3" → "903" — handles mixed digit/word patterns
    # "nine oh three" stays as words — Gemini handles fully spoken numbers
    text = re.sub(r'(?<=\d)\s*\boh\b\s*(?=\d)', '0', text, flags=re.IGNORECASE)

    # Fix 4 — "word at known-domain" → "word@known-domain"
    # Only fires when word after "at" is a known email provider
    # Unknown domains: Gap 2 in _has_contact_info() already flagged for Gemini
    def replace_at(m):
        before = m.group(1)
        after  = m.group(2)
        if after.lower().split('.')[0] in _EMAIL_CONTEXT:
            return f"{before}@{after}"
        return m.group(0)

    text = re.sub(r'(\w+)\s+at\s+(\w+(?:\.\w+)*)', replace_at, text)

    return text


# ── Gemini correction prompt ──────────────────────────────────────────────────

_CORRECTION_PROMPT = """You are a transcript corrector for a customer sales call.

The transcript was produced by speech-to-text and may contain errors in:
- Email addresses spoken out loud
- Phone numbers spoken as individual digits
- Spoken symbols like "at", "dot", "underscore", "hyphen"
- "oh" spoken instead of zero in phone numbers
- Letter spelling like "double u" meaning the letter W
- Minor grammatical errors introduced by speech-to-text
- Missing or incorrect punctuation

Examples:
  Input:  "my email is john underscore doe at gmail dot com"
  Output: "my email is john_doe@gmail.com"

  Input:  "call me on nine eight double two one three four five six seven"
  Output: "call me on 9822134567"

  Input:  "reach me at abhijit double u sharma at tatacloud dot co dot in"
  Output: "reach me at abhijitWsharma@tatacloud.co.in"

  Input:  "my number is nine oh three two one four five six seven eight"
  Output: "my number is 9032145678"

  Input:  "nine eight seven six five four three two one zero"
  Output: "9876543210"

  Input:  "john 2 3 at gmail dot com"
  Output: "john23@gmail.com"

  Input:  "98 seven six five four three two one zero"
  Output: "9876543210"

  Input:  "i wants to know the price of linux vm"
  Output: "I want to know the price of Linux VM."

  Input:  "yes i am looking for 128 vcpu and 1tb ram"
  Output: "Yes, I am looking for 128 vCPU and 1TB RAM."

Rules:
- Fix email addresses, phone numbers, and spoken symbols
- Fix grammatical errors (wrong verb forms, missing articles, incorrect tense) introduced by speech-to-text
- Fix punctuation — add capitalisation at sentence start, periods at sentence end, commas where naturally spoken
- Stay as close to the original wording as possible — do NOT rephrase, reword, or restructure
- Do NOT change the meaning of what the speaker said under any circumstances
- Do NOT add words that were not spoken — only fix what is clearly wrong
- Do NOT change product names, technical terms, company names, or pricing figures
- Do NOT explain anything — return ONLY the corrected transcript
- If nothing needs fixing return the transcript exactly as given

Transcript to correct:
{transcript}"""


# ── Main entry point ──────────────────────────────────────────────────────────

async def correct_transcript(text: str, session_id: str = "") -> str:
    """
    Two-stage transcript correction.

    Enforced order — critical for correctness:
      Stage 0: Empty check    — skip if nothing to correct
      Stage 1: _has_contact_info on ORIGINAL text — MUST run before regex
      Stage 2: Regex fixes    — always runs, zero latency, zero cost
      Stage 3: Gemini fix     — runs only when Stage 1 detected contact info

    Args:
        text:       Raw transcript from STT
        session_id: For log tracing

    Returns:
        Corrected transcript string. Never raises — falls back to
        regex-fixed text if Gemini fails.
    """

    # Stage 0 — skip empty
    if not text.strip():
        transcript_logger.debug(f"[{session_id}] Empty transcript — skipping correction")
        return text

    # Stage 1 — detect on ORIGINAL text before any regex changes
    # Critical: if regex runs first it converts "at" → "@" and
    # _has_contact_info misses it — Gemini never runs to fix regex errors
    needs_correction = _has_contact_info(text)

    # Stage 2 — regex always runs
    fixed = _regex_fix(text)
    if fixed != text:
        transcript_logger.debug(
            f"[{session_id}] Regex fix applied — before: \"{text}\" — after: \"{fixed}\""
        )
    else:
        transcript_logger.debug(f"[{session_id}] Regex fix — no changes")

    # Stage 3 — Gemini only if contact info detected in Stage 1
    if not needs_correction:
        transcript_logger.debug(f"[{session_id}] No contact info detected — skipping Gemini")
        return fixed

    transcript_logger.debug(f"[{session_id}] Contact info detected — running Gemini correction")

    try:
        prompt    = _CORRECTION_PROMPT.format(transcript=fixed)
        response  = await _correction_model.generate_content_async(prompt)
        corrected = response.text.strip()

        if corrected != fixed:
            transcript_logger.info(
                f"[{session_id}] Gemini correction applied\n"
                f"  before: \"{fixed}\"\n"
                f"  after:  \"{corrected}\""
            )
        else:
            transcript_logger.debug(f"[{session_id}] Gemini correction — no changes needed")

        return corrected

    except Exception as e:
        transcript_logger.error(
            f"[{session_id}] Gemini correction failed — {e} — using regex-fixed text"
        )
        return fixed