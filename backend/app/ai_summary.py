"""AI forensic summarization via Google Gemini.

This module generates a concise plain-language summary of detector evidence.
It is deliberately constrained so the model:
  - Only references signals present in the structured evidence JSON.
  - Uses hedged language ("detectors indicate", "signals suggest").
  - Never declares the media "real" or "fake" outright.
  - Never recommends actions (those come from fuse_evidence()).

If the API key is absent or the call fails the function returns None and a
warning is logged.  The scan always completes regardless.
"""

from __future__ import annotations

import json
import logging
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
- Do NOT recommend actions — those come from the system separately.
- Be concise: 2-3 sentences maximum.
"""


def _build_user_message(evidence: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
    """Serialise only the fields the model needs — strips large binary/hash data and raw percentages."""
    slim_evidence = []
    for item in evidence:
        slim_evidence.append({
            "label": item.get("label"),
            "source": item.get("source"),
            "risk": item.get("risk"),
            "details": {
                k: v for k, v in (item.get("details") or {}).items()
                if k not in ("sha256", "exif", "per_frame_scores", "image_model", "confidence", "triage", "review_priority")
            },
        })
    payload = {
        "evidence": slim_evidence,
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
    from .config import GEMINI_API_KEY

    key = api_key or GEMINI_API_KEY
    if not key:
        logger.debug("GEMINI_API_KEY not set — skipping AI summary.")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=_build_user_message(evidence, assessment),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        summary = response.text.strip() if response.text else None
        if summary:
            logger.info("AI summary generated (%d chars).", len(summary))
        return summary
    except Exception as exc:
        logger.warning("AI summary generation failed: %s", exc)
        return None
