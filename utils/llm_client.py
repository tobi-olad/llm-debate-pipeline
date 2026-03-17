# =============================================================================
# utils/llm_client.py — Three-server LLM client
# =============================================================================
#
# role="debater_a"  →  Llama 3.1 8B     (Server 1, open internet)
# role="debater_b"  →  GPT OSS 20B      (Server 3, UTSA VPN required)
# role="judge"      →  Qwen3-8B         (Server 2, open internet)
# role="debater"    →  alias for debater_a (used by baselines)
#
# NOTE: If Debater B calls fail, make sure you are connected to UTSA VPN.
# =============================================================================

import time
from openai import OpenAI
import config

# Lazy singletons — created on first use
_client_a    = None
_client_b    = None
_client_judge = None


def _get_client_a() -> OpenAI:
    global _client_a
    if _client_a is None:
        _client_a = OpenAI(api_key=config.DEBATER_A_API_KEY,
                           base_url=config.DEBATER_A_BASE_URL)
    return _client_a


def _get_client_b() -> OpenAI:
    global _client_b
    if _client_b is None:
        _client_b = OpenAI(api_key=config.DEBATER_B_API_KEY,
                           base_url=config.DEBATER_B_BASE_URL)
    return _client_b


def _get_client_judge() -> OpenAI:
    global _client_judge
    if _client_judge is None:
        _client_judge = OpenAI(api_key=config.JUDGE_API_KEY,
                               base_url=config.JUDGE_BASE_URL)
    return _client_judge


def call_llm(prompt: str, role: str = "debater_a", temperature: float = None) -> str:
    """
    Send a prompt to the appropriate server and return the response text.

    role : "debater_a" | "debater" → Llama 3.1 8B  (Server 1)
           "debater_b"             → GPT OSS 20B    (Server 3, needs VPN)
           "judge"                 → Qwen3-8B       (Server 2)
    """
    if role == "judge":
        client   = _get_client_judge()
        model    = config.JUDGE_MODEL
        temp     = temperature if temperature is not None else config.JUDGE_TEMP
        max_toks = config.JUDGE_MAX_TOKENS

    elif role == "debater_b":
        client   = _get_client_b()
        model    = config.DEBATER_B_MODEL
        temp     = temperature if temperature is not None else config.DEBATER_B_TEMP
        max_toks = config.MAX_TOKENS

    else:  # "debater_a" or "debater" (baselines)
        client   = _get_client_a()
        model    = config.DEBATER_A_MODEL
        temp     = temperature if temperature is not None else config.DEBATER_A_TEMP
        max_toks = config.MAX_TOKENS

    system = "You are a helpful, precise assistant. Follow the output format exactly."

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temp,
                max_tokens=max_toks,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(
                    f"LLM call failed for role='{role}' after 3 attempts: {e}\n"
                    + ("Debater B uses UTSA VPN — are you connected?" if role == "debater_b" else "")
                ) from e
            wait = 2 ** attempt
            print(f"  [LLM:{role}] Attempt {attempt + 1} failed ({e}), retrying in {wait}s…")
            time.sleep(wait)