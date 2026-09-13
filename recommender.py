"""
recommender.py
--------------
Pure Python. No Streamlit. Run this directly to test:

    python recommender.py

IMPORTANT — read this if you're confused about "it runs once and stops":
    This file has TWO lives.
    1. `python recommender.py`      → runs the STANDALONE TEST at the bottom,
                                      prints one recommendation, then EXITS.
                                      That exit is BY DESIGN — it's a one-shot test.
    2. `streamlit run app.py`       → app.py IMPORTS this file. The test block
                                      does NOT run (it is guarded by
                                      `if __name__ == "__main__"`). The Streamlit
                                      server stays alive in terminal 1 the whole time.

    So: keep `streamlit run app.py` running in terminal 1 (the app lives in the
    browser), and only run `python recommender.py` in a SEPARATE terminal when
    you want to test the logic headless.

Python logs appear in the TERMINAL — both when run standalone
AND when called from Streamlit (watch your terminal, not the browser).
"""

import json
import os
import re
import logging
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prompts import SYSTEM_PROMPT

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── API key ───────────────────────────────────────────────────────────────────
load_dotenv()

def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file "
            "(get a free key at aistudio.google.com)."
        )
    return key

MODEL_NAME = "gemini-3-flash-preview"   # free tier, fast, 1M context


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(filepath: str = "models.json") -> dict:
    """Load models.json from disk. Streamlit caches this after first call."""
    abs_path = os.path.abspath(filepath)
    log.info("Loading models from: %s", abs_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = sum(len(v) for v in data["models"].values())
    log.info("Loaded %d models across %d categories", total, len(data["models"]))
    return data


# ── Slim context ──────────────────────────────────────────────────────────────
def build_slim_context(models_data: dict) -> list:
    """
    Strip nested/heavy fields before injecting into prompt.
    Reduces token usage significantly.
    """
    all_models = []
    for category_models in models_data["models"].values():
        for m in category_models:
            slim = {
                "id":               m["id"],
                "name":             m["name"],
                "provider":         m["provider"],
                "category":         m["category"],
                "tags":             m["tags"],
                "description":      m["description"],
                "context_window":   m["context_window"],
                "pricing":          m["pricing"],
                "strengths":        m["strengths"],
                "weaknesses":       m["weaknesses"],
                "best_for":         m["best_for"],
                "not_ideal_for":    m["not_ideal_for"],
                "license":          m["license"],
                "api_model_string": m["availability"].get("api_model_string"),
                "self_host":        m["availability"].get("self_host", False),
            }
            if "modalities" in m:
                slim["modalities"] = m["modalities"]
            all_models.append(slim)

    log.info(
        "Slim context built: %d models, %d chars (~%d tokens)",
        len(all_models),
        len(json.dumps(all_models)),
        len(json.dumps(all_models)) // 4,
    )
    return all_models


# ── JSON extractor ────────────────────────────────────────────────────────────
def extract_json(text: str) -> str:
    """
    Gemini occasionally wraps JSON in ```json fences or adds a preamble.
    This strips all of that and returns the raw JSON string.
    """
    text = text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Strip any text before the first {
    brace_start = text.find("{")
    if brace_start > 0:
        log.warning("Stripping %d chars of preamble before JSON", brace_start)
        text = text[brace_start:]

    # Strip any text after the last }
    brace_end = text.rfind("}")
    if brace_end != -1 and brace_end < len(text) - 1:
        log.warning("Stripping trailing content after JSON")
        text = text[: brace_end + 1]

    return text.strip()


# ── Main function ─────────────────────────────────────────────────────────────
def get_recommendation(use_case: str, models_data: dict) -> dict:
    """
    Takes a plain-English use case + loaded models dict.

    Returns one of:
        {"success": True,  "data": { ...recommendation dict... }}
        {"success": False, "error": "human-readable error message"}
    """
    log.info("=" * 55)
    log.info("New recommendation request")
    log.info("=" * 55)
    log.info("Use case (%d chars): %.120s", len(use_case), use_case)

    # Validate input
    use_case = use_case.strip()
    if len(use_case) < 10:
        return {"success": False, "error": "Please describe your use case in more detail."}
    if len(use_case) > 2000:
        return {"success": False, "error": "Use case too long. Keep it under 2000 characters."}

    # Build prompt content
    slim_models = build_slim_context(models_data)
    user_message = (
        f"Here is the model database:\n{json.dumps(slim_models, indent=2)}\n\n"
        f'User\'s use case:\n"{use_case}"\n\n'
        f"Analyze the use case and return ONLY a raw JSON object. No markdown, no preamble."
    )
    log.info("Prompt built — total chars: %d", len(user_message))

    # Resolve API key (raises EnvironmentError if missing → caught below)
    try:
        api_key = _get_api_key()
    except EnvironmentError as e:
        return {"success": False, "error": str(e)}

    # Call Gemini
    try:
        log.info("Calling Gemini: %s", MODEL_NAME)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=8000,
            ),
        )
        log.info("Gemini call successful")
    except Exception as e:
        err = str(e)
        log.error("Gemini API error: %s", err)
        if "API_KEY" in err.upper() or "INVALID" in err.upper() or "403" in err:
            return {"success": False, "error": "Invalid API key. Double-check your GEMINI_API_KEY in .env."}
        elif "429" in err or "QUOTA" in err.upper() or "RATE" in err.upper():
            return {"success": False, "error": "Free tier rate limit hit (60 req/min). Wait a minute and retry."}
        elif "SAFETY" in err.upper():
            return {"success": False, "error": "Request blocked by Gemini safety filters. Try rephrasing your use case."}
        else:
            return {"success": False, "error": f"Gemini API error: {err}"}

    # Extract text from response.
    # NOTE: response.text can be None (e.g. safety block with no candidates).
    # Guard it — previously len(raw_text) crashed with TypeError here.
    raw_text = response.text
    if not raw_text:
        log.error("Gemini returned empty text. prompt_feedback: %s",
                  getattr(response, "prompt_feedback", None))
        return {
            "success": False,
            "error": (
                "Gemini returned an empty response (usually a safety block or "
                "filter). Try rephrasing your use case."
            ),
        }
    log.info("Raw response length: %d chars", len(raw_text))
    log.info("Raw response (first 400 chars):\n%s", raw_text[:400])

    # Clean and parse
    clean_text = extract_json(raw_text)
    log.info("Cleaned JSON (first 200 chars):\n%s", clean_text[:200])

    try:
        recommendation = json.loads(clean_text)
        log.info("JSON parsed successfully. Top-level keys: %s", list(recommendation.keys()))

        # Warn on missing expected keys
        for key in ["best_pick", "alternatives", "use_case_summary", "not_recommended"]:
            if key not in recommendation:
                log.warning("Expected key missing from response: '%s'", key)

        return {"success": True, "data": recommendation}

    except json.JSONDecodeError as e:
        log.error("JSON parse failed: %s", e)
        log.error("Full cleaned text:\n%s", clean_text)
        return {
            "success": False,
            "error": (
                "Gemini returned a response that couldn't be parsed as JSON. "
                "Try rephrasing your use case or running again.\n"
                f"(Parse error: {e})"
            ),
        }
