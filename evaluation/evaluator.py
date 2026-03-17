# =============================================================================
# evaluation/evaluator.py — Baselines (Direct QA, Self-Consistency) + metrics
# =============================================================================

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import config
from utils.llm_client import call_llm
from utils.parsing import fmt_choices, extract_letter


# ── Direct QA baseline ────────────────────────────────────────────────────────

def direct_qa(question: str, choices: list[str]) -> Optional[str]:
    """Single CoT answer. Returns the predicted answer letter."""
    template = (Path(__file__).parent.parent / "prompts" / "direct_qa.txt").read_text()
    prompt   = template.format(question=question, choices=fmt_choices(choices))
    raw      = call_llm(prompt, role="debater")
    return extract_letter(raw, choices)


# ── Self-Consistency baseline ─────────────────────────────────────────────────

def self_consistency(question: str, choices: list[str]) -> Optional[str]:
    """
    Sample SC_N independent answers at high temperature; return majority vote.
    Uses the same debater server as the baseline.
    """
    template = (Path(__file__).parent.parent / "prompts" / "direct_qa.txt").read_text()
    prompt   = template.format(question=question, choices=fmt_choices(choices))

    answers = []
    for _ in range(config.SC_N):
        raw = call_llm(prompt, role="debater", temperature=config.SC_TEMP)
        answers.append(extract_letter(raw, choices))

    valid = [a for a in answers if a]
    return Counter(valid).most_common(1)[0][0] if valid else None


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def accuracy(results: list[dict], pred_key: str) -> tuple[float, int, int]:
    """
    Compute accuracy over a list of result dicts.
    pred_key : the key holding the predicted letter (e.g. "verdict", "direct", "sc")
    Returns  : (accuracy_float, n_correct, n_total_with_prediction)
    """
    valid   = [r for r in results if r.get(pred_key)]
    if not valid:
        return 0.0, 0, 0
    correct = sum(1 for r in valid if r.get(pred_key) == r["ground_truth"])
    return correct / len(valid), correct, len(valid)


# ── Results loading ───────────────────────────────────────────────────────────

def load_debate_logs() -> list[dict]:
    """Load all per-question JSON logs written by the orchestrator."""
    log_dir = Path(config.LOG_DIR) / "debate_logs"
    logs = []
    for p in sorted(log_dir.glob("question_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(p) as f:
            logs.append(json.load(f))
    return logs


def load_baseline_logs() -> tuple[list[dict], list[dict]]:
    """Load direct QA and self-consistency result lists."""
    d_path = Path(config.LOG_DIR) / "debate_logs" / "direct_results.json"
    s_path = Path(config.LOG_DIR) / "debate_logs" / "sc_results.json"
    direct = json.load(open(d_path)) if d_path.exists() else []
    sc     = json.load(open(s_path)) if s_path.exists() else []
    return direct, sc


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(debate_res: list[dict],
                        direct_res: list[dict],
                        sc_res:     list[dict]) -> None:
    d_acc, d_cor, d_tot = accuracy(debate_res, "verdict")
    q_acc, q_cor, q_tot = accuracy(direct_res, "direct")
    s_acc, s_cor, s_tot = accuracy(sc_res,     "sc")

    print("\n" + "=" * 52)
    print(f"{'Method':<22} {'Accuracy':>9} {'Correct':>8} {'N':>5}")
    print("-" * 52)
    print(f"{'Debate + Judge':<22} {d_acc:>8.1%} {d_cor:>8} {d_tot:>5}")
    print(f"{'Direct QA (CoT)':<22} {q_acc:>8.1%} {q_cor:>8} {q_tot:>5}")
    print(f"{'Self-Consistency':<22} {s_acc:>8.1%} {s_cor:>8} {s_tot:>5}")
    print("=" * 52)


def print_confidence_breakdown(debate_res: list[dict]) -> None:
    print("\nDebate accuracy by judge confidence:")
    for conf in range(1, 6):
        subset = [r for r in debate_res if r.get("confidence") == conf]
        if subset:
            acc, cor, tot = accuracy(subset, "verdict")
            print(f"  Confidence {conf}: {acc:.1%}  (n={tot})")