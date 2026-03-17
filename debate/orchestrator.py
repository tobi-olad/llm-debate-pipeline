# =============================================================================
# debate/orchestrator.py — 4-phase debate loop controller
# =============================================================================

import json
from pathlib import Path

import config
from agents import debaters, judge
from utils.parsing import build_transcript, extract_letter, letter_to_text


def run(qid: str, question: str, choices: list[str], ground_truth: str, question_num: int = None) -> dict:
    letters = "ABCD"[: len(choices)]

    result = {
        "id": qid, "question": question,
        "choices": choices, "ground_truth": ground_truth,
    }

    print(f"\n  Q: {question}")

    # ── Phase 1: Both debaters independently pick an initial position ─────────
    pos_a = debaters.get_initial_position(question, choices, role="debater_a")
    pos_b = debaters.get_initial_position(question, choices, role="debater_b")

    result["init_a"] = pos_a
    result["init_b"] = pos_b

    ans_a = pos_a["answer"] or letters[0]
    ans_b = pos_b["answer"] or letters[1]

    print(f"  → Debater A initial pick : {letter_to_text(ans_a, choices)}")
    print(f"  → Debater B initial pick : {letter_to_text(ans_b, choices)}")

    # ── Consensus check ───────────────────────────────────────────────────────
    if ans_a == ans_b:
        print(f"  → Consensus reached — skipping debate")
        result.update({
            "consensus":  True,
            "verdict":    ans_a,
            "rounds":     [],
            "early_stop": False,
            "judge":      {},
        })

    else:
        print(f"  → No consensus — starting debate")

        # ── Phase 2: Multi-round debate ───────────────────────────────────────
        rounds, early_stop = _run_debate(question, choices, ans_a, ans_b)
        result["rounds"]     = rounds
        result["early_stop"] = early_stop
        result["consensus"]  = False

        # ── Phase 3: Judge ────────────────────────────────────────────────────
        print(f"\n  ── Judge evaluating transcript ──")
        full_transcript = build_transcript(rounds)
        judge_out       = judge.evaluate(question, choices, full_transcript)
        result["judge"]      = judge_out
        result["verdict"]    = judge_out["verdict"]
        result["confidence"] = judge_out["confidence"]

        print(f"  → Winner   : Debater {judge_out.get('winner', '?')}")
        print(f"  → Confidence : {judge_out.get('confidence', '?')}/5")
        print(f"  → Judge reasoning : {judge_out.get('reasoning', '')}")

    # ── Phase 4: Score and log ────────────────────────────────────────────────
    result["correct"] = (result.get("verdict") == ground_truth)

    verdict_txt = letter_to_text(result.get("verdict"), choices)
    gt_txt      = letter_to_text(ground_truth, choices)
    print(f"\n  {'─' * 56}")
    print(f"  → Verdict : {verdict_txt}")
    print(f"  → Correct : {gt_txt}  {'Good' if result['correct'] else 'Wrong'}")
    print(f"  {'─' * 56}")

    log_dir = Path(config.LOG_DIR) / "debate_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"question_{question_num}.json" if question_num is not None else f"{qid}.json"
    with open(log_dir / filename, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_debate(question: str, choices: list[str],
                ans_a: str, ans_b: str) -> tuple[list[dict], bool]:
    """Run up to MAX_ROUNDS of debate. Returns (rounds, early_stopped)."""
    rounds            = []
    consecutive_agree = 0

    for round_num in range(1, config.MAX_ROUNDS + 1):
        print(f"\n  ── Round {round_num} of {config.MAX_ROUNDS} ──────────────────────────────────")
        transcript = build_transcript(rounds)

        # Debater A argues
        arg_a = debaters.argue_a(question, choices, ans_a, transcript, round_num)
        print(f"\n  [Debater A — {letter_to_text(ans_a, choices)}]")
        print(_indent(arg_a))

        # Debater B sees A's argument then responds
        transcript_with_a = (
            transcript
            + f"\n=== ROUND {round_num} ===\n[Debater A]\n{arg_a}\n"
        )
        arg_b = debaters.argue_b(question, choices, ans_a, ans_b,
                                transcript_with_a, round_num)
        print(f"\n  [Debater B — {letter_to_text(ans_b, choices)}]")
        print(_indent(arg_b))

        cur_a = extract_letter(arg_a, choices) or ans_a
        cur_b = extract_letter(arg_b, choices) or ans_b

        print(f"\n  → End of round {round_num} positions — "
              f"A: {letter_to_text(cur_a, choices)}  |  "
              f"B: {letter_to_text(cur_b, choices)}")

        rounds.append({
            "round": round_num,
            "arg_a": arg_a, "arg_b": arg_b,
            "pos_a": cur_a, "pos_b": cur_b,
        })

        # Adaptive early stop
        if cur_a == cur_b:
            consecutive_agree += 1
            if consecutive_agree >= config.EARLY_STOP and round_num >= 2:
                agreed_txt = letter_to_text(cur_a, choices)
                print(f"\nEarly stop — both converged on {agreed_txt} "
                      f"for {consecutive_agree} consecutive rounds")
                return rounds, True
        else:
            consecutive_agree = 0

    return rounds, False


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent every line of a block of text for readable console output."""
    return "\n".join(prefix + line for line in text.strip().splitlines())