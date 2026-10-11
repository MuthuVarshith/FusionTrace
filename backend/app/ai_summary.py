"""AI forensic summarization via Google Gemini (optional, opt-in per scan).

This module generates a concise plain-language summary of detector evidence.
It is deliberately constrained so the model:
  - Only references signals present in the structured evidence JSON.
  - Uses hedged language ("detectors indicate", "signals suggest").
  - Never declares the media "real" or "fake" outright.
  - Never recommends actions (those come from fuse_evidence()).

Privacy: only detector outputs are sent (signal names, model names, risk
scores, predictions and numeric detector details). The media file, its hash,
filename, EXIF and other file metadata never leave the server. Summaries run
only when the uploader ticks the opt-in box for that scan and the operator has
set ``GEMINI_API_KEY``; the model id comes from ``GEMINI_MODEL``.

If the API key is absent or the call fails the function returns None and a
warning is logged. The scan always completes regardless.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger("fusiontrace")

_SYSTEM_PROMPT = """\
You are a forensic analysis assistant for FusionTrace, a media authenticity tool.
You will be given structured JSON output from automated detectors that analysed a piece of media.
Your task is to write a 2-3 sentence plain-language summary of what the detectors found.

Rules you must follow:
- Only reference information that is present in the detector evidence provided.
- Do NOT make claims of authenticity or inauthenticity beyond what the detectors state.
- Do NOT say the media "is" real or fake. Use hedged language: "the detectors indicate", "signals suggest", etc.
- Do NOT include raw confidence percentage numbers (e.g., do not write 99.99% or 89.9%) or triage notes like "requires manual review". Focus purely on the nature of the detected signal.
- Signals whose status is not "ok" did not produce a score; mention them only as unavailable, if at all.
- Signals marked "heuristic": true are rule-based checks, not trained models; describe them as heuristic checks.
- Do NOT recommend actions — those come from the system separately.
- Be concise: 2-3 sentences maximum.
"""

# Evidence items derived from the file itself rather than from a detector.
_EXCLUDED_SOURCES = {"file-inspection", "image-metadata"}


@lru_cache(maxsize=1)
def _sdk_installed() -> bool:
    try:
        return importlib.util.find_spec("google.genai") is not None
    except (ImportError, ValueError):
        return False


def summaries_available() -> bool:
    """True when the operator configured a Gemini key and the SDK is installed."""
    from . import config as settings

    return bool(settings.GEMINI_API_KEY) and _sdk_installed()


def _slim_item(item: dict[str, Any]) -> dict[str, Any]:
    model = item.get("model") or {}
    details = item.get("details") or {}
    risk = item.get("risk")
    status = item.get("status", "ok")
    slim: dict[str, Any] = {
        "signal": item.get("label"),
        "source": item.get("source"),
        "status": status,
        "risk": round(float(risk), 3) if risk is not None else None,
        "weight": item.get("weight"),
        "model": model.get("display_name"),
        "heuristic": bool(model.get("heuristic")),
    }
    if details.get("prediction") in ("Fake", "Real"):
        slim["prediction"] = details["prediction"]
    if status != "ok" and isinstance(details.get("reason"), str):
        slim["reason"] = details["reason"]
    # Numeric detector outputs only (counts, scores); no strings, lists or nested data.
    numeric = {key: value for key, value in details.items() if isinstance(value, (int, float)) and key != "confidence"}
    if numeric:
        slim["details"] = numeric
    return slim


def _build_user_message(evidence: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
    """Serialise only detector outputs; nothing derived from the raw file is included."""
    payload = {
        "evidence": [_slim_item(item) for item in evidence if item.get("source") not in _EXCLUDED_SOURCES],
        "assessment": {
            "risk_score": assessment.get("risk_score"),
            "risk_rating": assessment.get("risk_rating"),
        },
    }
    return (
        "Here is the structured detector evidence for this media scan. "
        "Please write a plain-language forensic summary.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )


def generate_summary(
    evidence: list[dict[str, Any]],
    assessment: dict[str, Any],
    api_key: str | None = None,
) -> str | None:
    """Return a 2-3 sentence AI forensic summary, or None if unavailable."""
    from . import config as settings

    key = api_key or settings.GEMINI_API_KEY
    if not key:
        logger.debug("GEMINI_API_KEY not set — skipping AI summary.")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=_build_user_message(evidence, assessment),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        summary = response.text.strip() if response.text else None
        if summary:
            logger.info("AI summary generated with %s (%d chars).", settings.GEMINI_MODEL, len(summary))
        return summary
    except Exception as exc:  # noqa: BLE001 - any SDK/network failure just means no summary
        logger.warning("AI summary generation failed (%s): %s", settings.GEMINI_MODEL, exc)
        return None
