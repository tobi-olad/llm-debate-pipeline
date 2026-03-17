#!/usr/bin/env python3
# =============================================================================
# main.py — CLI entrypoint for the LLM Debate + Judge Pipeline
# =============================================================================
#
# Usage:
#   python main.py                    # run on default number of questions
#   python main.py --questions 100    # full experiment
#   python main.py --demo             # interactive single-question demo
#   python main.py --eval             # re-print results from saved logs
# =============================================================================

import argparse
import json
from pathlib import Path

import config
from debate import orchestrator
from data import arc_loader
from evaluation import evaluator


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_run(n: int) -> None:
    questions = arc_loader.load(n)
    print(f"\n[Run] {len(questions)} questions")
    print(f"      Debater A : {config.DEBATER_A_MODEL}")
    print(f"      Debater B : {config.DEBATER_B_MODEL}")
    print(f"      Judge    : {config.JUDGE_MODEL}\n")

    debate_res, direct_res, sc_res = [], [], []

    for i, q in enumerate(questions):
        print(f"[{i + 1}/{len(questions)}]", end="")

        # ── Debate pipeline ──
        res = orchestrator.run(q["id"], q["question"], q["choices"], q["answer"], question_num=i+1)
        debate_res.append(res)

        # ── Direct QA baseline ──
        dqa = evaluator.direct_qa(q["question"], q["choices"])
        direct_res.append({
            "question":q["question"],
            "ground_truth": q["answer"],
            "direct":       dqa,
            "correct":      dqa == q["answer"],
        })

        # ── Self-Consistency baseline ──
        sc = evaluator.self_consistency(q["question"], q["choices"])
        sc_res.append({
            "question":q["question"],
            "ground_truth": q["answer"],
            "sc":           sc,
            "correct":      sc == q["answer"],
        })

    # Save baseline logs
    log_dir = Path(config.LOG_DIR) / "debate_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    json.dump(direct_res, open(log_dir / "direct_results.json", "w"), indent=2)
    json.dump(sc_res,     open(log_dir / "sc_results.json",     "w"), indent=2)

    evaluator.print_summary_table(debate_res, direct_res, sc_res)


def cmd_eval() -> None:
    debate_logs          = evaluator.load_debate_logs()
    direct_logs, sc_logs = evaluator.load_baseline_logs()

    if not debate_logs:
        print("No debate logs found. Run: python main.py --questions 10")
        return
    if not direct_logs or not sc_logs:
        print("Baseline logs missing. Run the full pipeline first.")
        return

    evaluator.print_summary_table(debate_logs, direct_logs, sc_logs)
    evaluator.print_confidence_breakdown(debate_logs)


def cmd_demo() -> None:
    question = input("Enter your question: ").strip()
    print("Enter up to 4 answer choices (press Enter on an empty line to finish):")
    choices = []
    for i in range(4):
        c = input(f"  {chr(65 + i)}) ").strip()
        if not c:
            break
        choices.append(c)
    if len(choices) < 2:
        choices = ["True", "False", "Cannot be determined", "Not enough info"]
    answer = input(f"Correct answer letter (A–{chr(64 + len(choices))}): ").strip().upper() or "A"
    orchestrator.run("demo", question, choices, answer)


# ── Argument parser ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Debate + Judge Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                   # quick test (10 questions)\n"
            "  python main.py --questions 100   # full experiment\n"
            "  python main.py --demo            # interactive demo\n"
            "  python main.py --eval            # show results from saved logs\n"
        ),
    )
    parser.add_argument(
        "--questions", type=int, default=config.NUM_QUESTIONS,
        help=f"Number of questions to evaluate (default: {config.NUM_QUESTIONS})",
    )
    parser.add_argument("--demo", action="store_true", help="Run an interactive demo question")
    parser.add_argument("--eval", action="store_true", help="Re-print results from saved logs")
    args = parser.parse_args()

    if args.demo:
        cmd_demo()
    elif args.eval:
        cmd_eval()
    else:
        cmd_run(args.questions)


if __name__ == "__main__":
    main()