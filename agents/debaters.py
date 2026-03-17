# =============================================================================
# agents/debater.py — Debater A (Llama 3.1 8B) and Debater B (GPT OSS 20B)
# =============================================================================

from pathlib import Path
from utils.llm_client import call_llm
from utils.parsing import fmt_choices, extract_letter

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text()


def get_initial_position(question: str, choices: list[str],
                         role: str = "debater_a") -> dict:
    """
    Phase 1: Ask a debater to independently answer the question.
    role : "debater_a" → Llama 3.1 8B
           "debater_b" → GPT OSS 20B
    Returns {"raw": str, "answer": letter or None}
    """
    template = _load("direct_qa.txt")
    prompt   = template.format(question=question, choices=fmt_choices(choices))
    raw      = call_llm(prompt, role=role)
    return {"raw": raw, "answer": extract_letter(raw, choices)}


def argue_a(question: str, choices: list[str], answer: str,
            transcript: str, round_num: int) -> str:
    """Debater A (Llama 3.1 8B) produces one round of argument."""
    prompt = _load("debater_a.txt").format(
        question=question,
        choices=fmt_choices(choices),
        answer=answer,
        transcript=transcript or "(No previous rounds)",
        round_num=round_num,
    )
    return call_llm(prompt, role="debater_a")


def argue_b(question: str, choices: list[str], answer_a: str, answer_b: str,
            transcript: str, round_num: int) -> str:
    """Debater B (GPT OSS 20B) produces one round of counterargument."""
    prompt = _load("debater_b.txt").format(
        question=question,
        choices=fmt_choices(choices),
        answer_a=answer_a,
        answer_b=answer_b,
        transcript=transcript or "(No previous rounds)",
        round_num=round_num,
    )
    return call_llm(prompt, role="debater_b")