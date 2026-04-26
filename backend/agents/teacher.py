"""
teacher.py
Three independent LLM calls on the same question, each with a different
prompting style, run in parallel via asyncio.gather().
"""
import asyncio
import os
import re

from hf_client import hf_generate

_STYLE_PROMPTS = {
    "concise": (
        "Answer briefly in 1–2 sentences. "
        "End with: CONFIDENCE: X.X"
    ),
    "step_by_step": (
        "Solve step by step (max 5 steps). "
        "End with: CONFIDENCE: X.X"
    ),
    "devils_advocate": (
        "Give a wrong intuition briefly, then correct answer. "
        "End with: CONFIDENCE: X.X"
    ),
}


def _parse_confidence(text: str) -> float:
    """Extract CONFIDENCE value from last line; default 0.5."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        match = re.search(r"CONFIDENCE:\s*([01]?\.\d+|[01])", line)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    return 0.5


def _extract_reasoning(text: str, style: str) -> str:
    """
    Extract reasoning portion from the response.
    For step_by_step, return all step lines; for others return all but last line.
    """
    lines = text.strip().split("\n")
    # Remove the CONFIDENCE line
    body_lines = [l for l in lines if not l.strip().startswith("CONFIDENCE:")]
    if style == "step_by_step":
        return "\n".join(body_lines)
    # For concise / devils_advocate, the whole body is reasoning + answer
    return "\n".join(body_lines)


class TeacherPanel:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key
        self.model = model
        configured = os.environ.get("EVOAI_TEACHER_STYLES", "").strip()
        if configured:
            styles = [s.strip() for s in configured.split(",") if s.strip() in _STYLE_PROMPTS]
        else:
            low_tpm = os.environ.get("EVOAI_LOW_TPM_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
            styles = ["concise"] if low_tpm else ["concise", "step_by_step", "devils_advocate"]
        self.active_styles = styles or ["concise"]

    async def answer_all(self, question: str) -> list:
        """Run active teacher styles in parallel."""
        tasks = [
            self._call_teacher(question, style)
            for style in self.active_styles
        ]
        return list(await asyncio.gather(*tasks))

    async def _call_teacher(self, question: str, style: str) -> dict:
        """Call HF inference endpoint with a style-specific prompt; return structured dict."""
        system_prompt = _STYLE_PROMPTS[style]
        prompt = f"{system_prompt}\nQ: {question}"

        raw = await asyncio.to_thread(hf_generate, prompt, 0.7, 200)
        if not raw:
            raw = "Fallback answer due to HF failure"

        confidence = _parse_confidence(raw)
        reasoning = _extract_reasoning(raw, style)

        # For the "answer" field, use the first non-empty reasoning line
        lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
        answer = lines[0] if lines else raw.strip()

        return {
            "style": style,
            "answer": answer,
            "reasoning": reasoning,
            "confidence": confidence,
        }
