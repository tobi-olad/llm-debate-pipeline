# =============================================================================
# agents/judge.py — LLM Judge agent
# =============================================================================

import re
from pathlib import Path
from utils.llm_client import call_llm
from utils.parsing import fmt_choices, extract_letter, extract_section, extract_confidence, strip_think_tags

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def evaluate(question: str, choices: list[str], transcript: str) -> dict:
    """
    Read the full debate transcript and return a structured verdict dict:
        cot, strong_a, weak_a, strong_b, weak_b,
        winner, verdict (letter), confidence (1-5), reasoning
    """
    prompt = (PROMPT_DIR / "judge.txt").read_text().format(
        question=question,
        choices=fmt_choices(choices),
        transcript=transcript,
    )
    raw = call_llm(prompt, role="judge")
    raw = strip_think_tags(raw)  # Qwen3 emits <think>...</think> before output

    # Primary extraction
    verdict = extract_letter(raw, choices)

    # Fallback: look for bare VERDICT: C pattern
    if not verdict:
        m = re.search(r"VERDICT\s*:\s*([A-D])", raw, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()

    return {
        "raw":        raw,
        "cot":        extract_section(raw, "COT_ANALYSIS"),
        "strong_a":   extract_section(raw, "STRONGEST_ARGUMENT_A"),
        "weak_a":     extract_section(raw, "WEAKEST_ARGUMENT_A"),
        "strong_b":   extract_section(raw, "STRONGEST_ARGUMENT_B"),
        "weak_b":     extract_section(raw, "WEAKEST_ARGUMENT_B"),
        "winner":     extract_section(raw, "WINNER")[:2].strip(),
        "verdict":    verdict,
        "confidence": extract_confidence(raw),
        "reasoning":  extract_section(raw, "REASONING"),
    }