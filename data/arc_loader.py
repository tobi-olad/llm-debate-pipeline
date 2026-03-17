# =============================================================================
# datasets/arc_loader.py — ARC-Challenge dataset loader
# =============================================================================
# Downloads from HuggingFace on first run; falls back to a built-in sample
# if the `datasets` library is not installed or the network is unavailable.
# =============================================================================

import random
import config

# ── Built-in fallback (no internet needed) ────────────────────────────────────
_BUILTIN = [
    {"id": "q1", "question": "Which of the following is a primary color of light?",
     "choices": ["Red", "Brown", "Pink", "Black"], "answer": "A"},
    {"id": "q2", "question": "A student adds salt to water. What happens to the water?",
     "choices": ["Becomes lighter", "Becomes denser", "Becomes less dense", "Becomes a different substance"],
     "answer": "B"},
    {"id": "q3", "question": "Which object is the best conductor of electricity?",
     "choices": ["Wooden stick", "Rubber ball", "Copper wire", "Plastic ruler"], "answer": "C"},
    {"id": "q4", "question": "Why do whales breathe air instead of filtering oxygen from water?",
     "choices": ["Evolved from land mammals", "Cannot filter oxygen from water",
                 "Water lacks enough oxygen", "Need more oxygen than fish"], "answer": "A"},
    {"id": "q5", "question": "Which tool best measures the volume of a liquid?",
     "choices": ["Ruler", "Scale", "Thermometer", "Graduated cylinder"], "answer": "D"},
]


def load(n: int = config.NUM_QUESTIONS) -> list[dict]:
    """
    Return up to n questions as a list of dicts:
        { id, question, choices: list[str], answer: letter }
    """
    try:
        return _from_huggingface(n)
    except Exception as e:
        print(f"[data] HuggingFace unavailable ({e}). Using built-in sample.")
        return _from_builtin(n)


def _from_huggingface(n: int) -> list[dict]:
    from datasets import load_dataset
    ds  = load_dataset("ai2_arc", "ARC-Challenge", split=config.DATA_SPLIT)
    rng = random.Random(config.RANDOM_SEED)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idx:
        row    = ds[i]
        texts  = row["choices"]["text"]
        labels = row["choices"]["label"]
        key    = row["answerKey"]

        # Normalize numeric keys ("1" → "A")
        if key.isdigit():
            key = chr(64 + int(key))
        if labels[0].isdigit():
            labels = [chr(64 + int(l)) for l in labels]

        out.append({
            "id":       row["id"],
            "question": row["question"],
            "choices":  texts[:4],
            "answer":   key.upper(),
        })
    return out


def _from_builtin(n: int) -> list[dict]:
    rng  = random.Random(config.RANDOM_SEED)
    pool = _BUILTIN * (n // len(_BUILTIN) + 1)
    return rng.sample(pool, min(n, len(pool)))