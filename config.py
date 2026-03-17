# =============================================================================
# config.py — All settings in one place. Never hardcode values elsewhere.
# =============================================================================
# API keys are loaded from a .env file — never hardcoded here.
# Copy .env.example to .env and fill in your keys before running.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"  → Copy .env.example to .env and fill in your keys."
        )
    return val

# ── UTSA Server 1: Llama 3.1 8B ── Debater A + baselines ─────────────────────
DEBATER_A_API_KEY  = _require("DEBATER_A_API_KEY")
DEBATER_A_BASE_URL = "http://149.165.173.247:8888/v1"
DEBATER_A_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
DEBATER_A_TEMP     = 0.7

# ── UTSA Server 3: GPT OSS 20B ── Debater B  ──────────────
DEBATER_B_API_KEY  = _require("DEBATER_B_API_KEY")
DEBATER_B_BASE_URL = "http://10.100.1.212:8888/v1"
DEBATER_B_MODEL    = "openai/gpt-oss-20b"
DEBATER_B_TEMP     = 0.7

# ── UTSA Server 2: Qwen3-8B ── Judge ─────────────────────────────────────────
JUDGE_API_KEY      = _require("JUDGE_API_KEY")
JUDGE_BASE_URL     = "http://149.165.171.140:8888/v1"
JUDGE_MODEL        = "Qwen/Qwen3-8B"
JUDGE_TEMP         = 0.3

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS         = 1024
JUDGE_MAX_TOKENS   = 1536

# ── Debate protocol ───────────────────────────────────────────────────────────
MAX_ROUNDS         = 3
EARLY_STOP         = 2

# ── Baselines ─────────────────────────────────────────────────────────────────
SC_N               = 5
SC_TEMP            = 0.9

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_QUESTIONS      = 10
RANDOM_SEED        = 42
DATA_SPLIT         = "test"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR            = "logs"