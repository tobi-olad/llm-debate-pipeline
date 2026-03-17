# =============================================================================
# utils/parsing.py — All regex / text extraction helpers
# =============================================================================

import re
from typing import Optional


def strip_think_tags(text: str) -> str:
    """
    Qwen3 and some other models emit <think>...</think> blocks before their
    actual response. Strip them so downstream parsing sees clean output.
    """
    if not text:
        return text
    # Remove <think>...</think> blocks (including multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def fmt_choices(choices: list[str]) -> str:
    """Format choices as 'A) foo  B) bar  C) baz'"""
    return "  ".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))


def letter_to_text(letter: Optional[str], choices: list[str]) -> str:
    """Return e.g. 'B) endothermic' given letter='B' and the choices list."""
    if not letter:
        return "?"
    idx = ord(letter.upper()) - ord("A")
    if 0 <= idx < len(choices):
        return f"{letter}) {choices[idx]}"
    return letter


def extract_letter(text: str, choices: list[str]) -> Optional[str]:
    """
    Pull the chosen answer letter (A–D) from LLM output.
    Tries structured fields first, falls back to last standalone letter.
    """
    text = strip_think_tags(text)
    for field in ("ANSWER", "POSITION", "VERDICT"):
        m = re.search(rf"{field}\s*:\s*([A-D])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    letters = "ABCD"[: len(choices)]
    hits = re.findall(rf"\b([{letters}])\b", text.upper())
    return hits[-1] if hits else None


def extract_section(text: str, label: str) -> str:
    """Extract the content block under a labelled section heading."""
    text = strip_think_tags(text)
    pattern = rf"{label}\s*:\s*(.*?)(?=\n[A-Z_]{{3,}}\s*:|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_confidence(text: str) -> Optional[int]:
    """Extract the judge's confidence score (1–5)."""
    text = strip_think_tags(text)
    m = re.search(r"CONFIDENCE\s*:\s*([1-5])", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def build_transcript(rounds: list[dict]) -> str:
    """Build a human-readable debate transcript from round records."""
    lines = []
    for r in rounds:
        lines.append(f"=== ROUND {r['round']} ===")
        lines.append(f"[Debater A]\n{r['arg_a']}\n")
        lines.append(f"[Debater B]\n{r['arg_b']}\n")
    return "\n".join(lines)