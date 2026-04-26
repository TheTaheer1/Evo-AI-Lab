"""
critic.py
Evaluate reasoning quality of each teacher answer on three axes:
logical soundness, completeness, and absence of shortcuts/circular reasoning.
"""
import asyncio
import json
import os
import re

from hf_client import hf_generate

_HF_EMPTY_FALLBACK = (
    '{"gold_answer":"UNVERIFIABLE","teacher_labels":'
    '["unverifiable","unverifiable","unverifiable"]}'
)

_CRITIC_SYSTEM = (
    "Score reasoning (0–10 each): logical, completeness, no_shortcuts. "
    "Return JSON: {logical, completeness, no_shortcuts, overall}"
)


def _safe_parse_critic_json(raw: str) -> dict:
    """Parse critic JSON safely with fallback defaults."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            logical = int(parsed.get("logical", 5))
            completeness = int(parsed.get("completeness", 5))
            no_shortcuts = int(parsed.get("no_shortcuts", 5))
            overall = float(parsed.get("overall", (logical + completeness + no_shortcuts) / 3.0))
            flags = parsed.get("flags", [])
            if not isinstance(flags, list):
                flags = []
            return {
                "logical": logical,
                "completeness": completeness,
                "no_shortcuts": no_shortcuts,
                "flags": flags,
                "overall": round(overall, 2),
            }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"[Critic] JSON parse error: {e} | raw={raw[:120]!r}")
    # Safe defaults
    return {
        "logical": 5,
        "completeness": 5,
        "no_shortcuts": 5,
        "flags": ["parse_error"],
        "overall": 5.0,
    }


class Critic:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key
        self.model = model
        disable = os.environ.get("EVOAI_DISABLE_CRITIC", "false").strip().lower()
        self.disabled = disable in {"1", "true", "yes"}
        if self.disabled:
            print("[Critic] ⚠️  DISABLED via EVOAI_DISABLE_CRITIC — all reasoning scores will be 5.0")
        else:
            print("[Critic] ENABLED")

    async def evaluate_all(self, question: str, teacher_outputs: list) -> list:
        """Evaluate all teacher outputs in parallel using asyncio.gather()."""
        if self.disabled:
            return [
                {
                    **t,
                    "reasoning_score": 5.0,
                    "logical": 5,
                    "completeness": 5,
                    "no_shortcuts": 5,
                    "flags": ["critic_disabled"],
                    "critic_overall": 5.0,
                }
                for t in teacher_outputs
            ]
        tasks = [self._evaluate_one(question, t) for t in teacher_outputs]
        return list(await asyncio.gather(*tasks))

    async def _evaluate_one(self, question: str, teacher_output: dict) -> dict:
        """Score one teacher output on three reasoning axes; return augmented dict."""
        answer = teacher_output.get("answer", "")
        reasoning = teacher_output.get("reasoning", "")

        prompt = f"{_CRITIC_SYSTEM}\nQ: {question}\nA: {answer}\nR: {reasoning[:300]}"
        raw = await asyncio.to_thread(hf_generate, prompt, 0.3, 200)
        if not raw:
            print(
                f"[Critic] Empty HF response for style={teacher_output.get('style','?')} — using fallback JSON"
            )
            raw = _HF_EMPTY_FALLBACK
        scores = _safe_parse_critic_json(raw)

        # reasoning_score is average of the three axes
        reasoning_score = round(
            (scores["logical"] + scores["completeness"] + scores["no_shortcuts"]) / 3.0, 2
        )

        return {
            **teacher_output,
            "reasoning_score": reasoning_score,
            "logical": scores["logical"],
            "completeness": scores["completeness"],
            "no_shortcuts": scores["no_shortcuts"],
            "flags": scores["flags"],
            "critic_overall": scores["overall"],
        }
