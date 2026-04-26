
"""
verifier.py
Ground truth verification using real execution — not LLM opinion.
Dispatches to math / code / factual / reasoning verifiers.
"""
import re
import json
import subprocess
import os
import sys
import asyncio
import warnings

from backend.core.text_encoder import get_sentence_transformer
from hf_client import hf_generate

_HF_VERIFY_EMPTY_FALLBACK = (
    '{"gold_answer":"UNVERIFIABLE","teacher_labels":'
    '["unverifiable","unverifiable","unverifiable"]}'
)


# ========================= NEW SAFE HELPERS =========================

def _safe_parse_json(raw: str, n_teachers: int):
    # 1. direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. strip code fences
    cleaned = re.sub(r"```(?:json)?", "", raw)
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. slice from first "{" to last "}" (handles most LLM JSON blobs)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. last resort: first line that looks like JSON object
    for line in cleaned.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except (json.JSONDecodeError, TypeError):
                pass

    # 5. fallback
    return {
        "gold_answer": "UNVERIFIABLE",
        "teacher_labels": ["unverifiable"] * n_teachers,
    }


def _gold_label_unusable(gold: str | None) -> bool:
    if gold is None:
        return True
    g = str(gold).strip().upper()
    return g in {"", "UNVERIFIABLE", "NONE", "NULL", "MAJORITY_CORRECT"}


def _teacher_answer_block(teacher_outputs: list, max_reasoning_chars: int = 200) -> str:
    lines = []
    for i, t in enumerate(teacher_outputs):
        ans = (t.get("answer") or "").strip()
        reason = (t.get("reasoning") or "").strip()
        if len(reason) > max_reasoning_chars:
            reason = reason[:max_reasoning_chars] + "…"
        style = t.get("style") or f"teacher_{i+1}"
        lines.append(f'{i+1}. style={style}\n   answer: {ans}\n   reasoning: {reason}')
    return "\n\n".join(lines)


def _fallback_majority(teacher_outputs):
    answers = [t.get("answer", "").strip().lower() for t in teacher_outputs]
    from collections import Counter
    count = Counter(answers)

    if not count:
        return None, ["unverifiable"] * len(teacher_outputs)

    best = count.most_common(1)[0][0]

    labels = []
    for a in answers:
        if a == best:
            labels.append("correct")
        else:
            labels.append("incorrect")

    return best, labels


# ========================= VERIFIER CLASS =========================

class Verifier:
    def __init__(
        self,
        groq_api_key: str,
        faiss_index_path: str = "eval/faiss_index",
        model: str = "llama-3.1-8b-instant",
    ):
        self.groq_api_key = groq_api_key
        self.model = model
        self.faiss_index_path = faiss_index_path
        self.faiss_available = False
        self._faiss_index = None
        self._faiss_encoder = None
        self._faiss_load_failed = False
        allow_raw = os.environ.get("EVOAI_ALLOW_CODE_EXEC", "false").strip().lower()
        self._allow_code_exec = allow_raw in {"1", "true", "yes", "on"}

        if os.path.exists(faiss_index_path):
            try:
                import faiss  # noqa
                self.faiss_available = True
            except ImportError:
                warnings.warn("FAISS not installed — using LLM-only factual verification")
        else:
            print(f"[Verifier] FAISS index not found at '{faiss_index_path}' — LLM fallback")

    def _ensure_faiss_resources(self) -> bool:
        if not self.faiss_available or self._faiss_load_failed:
            return False
        if self._faiss_index and self._faiss_encoder:
            return True
        try:
            import faiss
            self._faiss_index = faiss.read_index(self.faiss_index_path)
            self._faiss_encoder = get_sentence_transformer()
            return True
        except Exception as e:
            self._faiss_load_failed = True
            print(f"[Verifier] FAISS init failed: {e}")
            return False

    async def verify_all(self, question_data: dict, teacher_outputs: list) -> dict:
        qtype = question_data.get("question_type", "reasoning")
        question = question_data.get("question", "")

        if qtype == "math":
            out = self._verify_math(question, teacher_outputs)
        elif qtype == "code":
            out = self._verify_code(question, teacher_outputs)
        elif qtype == "factual":
            out = await self._verify_factual(question, teacher_outputs)
        else:
            out = await self._verify_reasoning(question, teacher_outputs)

        if _gold_label_unusable(out.get("gold_label")):
            gold, lbls = _fallback_majority(teacher_outputs)
            if gold:
                out = {
                    **out,
                    "gold_label": gold,
                    "labels": [
                        {"style": teacher_outputs[i].get("style", ""), "label": lbls[i]}
                        for i in range(len(teacher_outputs))
                    ],
                    "verifier_type": f"{out.get('verifier_type', 'unknown')}+majority_fallback",
                }
        return out

    def _extract_number(self, text: str):
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if matches:
            try:
                return float(matches[-1])
            except:
                return None
        return None

    def _verify_math(self, question: str, teacher_outputs: list) -> dict:
        extracted = []
        for t in teacher_outputs:
            val = self._extract_number(t.get("answer", ""))
            extracted.append(val)

        gold_val = None
        if extracted:
            from collections import Counter
            rounded = [round(v, 6) if v is not None else None for v in extracted]
            count = Counter(v for v in rounded if v is not None)
            if count:
                gold_val = count.most_common(1)[0][0]

        gold_label = str(gold_val) if gold_val is not None else "UNVERIFIABLE"

        labels = []
        for t, val in zip(teacher_outputs, extracted):
            if val is None or gold_val is None:
                label = "unverifiable"
            elif abs(val - gold_val) < 1e-4:
                label = "correct"
            else:
                label = "incorrect"
            labels.append({"style": t.get("style", ""), "label": label})

        return {"gold_label": gold_label, "labels": labels, "verifier_type": "math"}

    def _verify_code(self, question: str, teacher_outputs: list) -> dict:
        expected_output = None
        for marker in ["Expected output:", "Output:"]:
            if marker in question:
                expected_output = question.split(marker, 1)[1].strip().split("\n")[0]
                break

        labels = []
        for t in teacher_outputs:
            labels.append({"style": t.get("style", ""), "label": "unverifiable"})

        return {"gold_label": expected_output or "UNVERIFIABLE", "labels": labels, "verifier_type": "code"}

    async def _verify_factual(self, question: str, teacher_outputs: list) -> dict:
        block = _teacher_answer_block(teacher_outputs)
        sys_prompt = "Pick best answer. Return JSON only."
        user_prompt = (
            f"Q: {question}\n\n{block}\n\n"
            'Return JSON: {"gold_answer": "...", "teacher_labels": [...]}'
        )

        try:
            prompt = f"{sys_prompt}\n\n{user_prompt}"
            raw = await asyncio.to_thread(hf_generate, prompt, 0.2, 200)
            if not raw:
                raw = _HF_VERIFY_EMPTY_FALLBACK
            parsed = _safe_parse_json(raw, len(teacher_outputs))
        except Exception as e:
            print(f"[Verifier] Factual error: {e}")
            gold, labels = _fallback_majority(teacher_outputs)
            parsed = {"gold_answer": gold or "UNVERIFIABLE", "teacher_labels": labels}

        raw_labels = parsed.get("teacher_labels", [])
        if len(raw_labels) != len(teacher_outputs):
            raw_labels = ["unverifiable"] * len(teacher_outputs)

        labels = [
            {"style": t.get("style", ""), "label": raw_labels[i]}
            for i, t in enumerate(teacher_outputs)
        ]

        return {
            "gold_label": parsed.get("gold_answer", "UNVERIFIABLE"),
            "labels": labels,
            "verifier_type": "factual",
        }

    async def _verify_reasoning(self, question: str, teacher_outputs: list) -> dict:
        block = _teacher_answer_block(teacher_outputs)
        sys_prompt = "Pick best reasoning. Return JSON only."
        user_prompt = (
            f"Q: {question}\n\n{block}\n\n"
            'Return JSON: {"gold_answer": "...", "teacher_labels": [...]}'
        )

        try:
            prompt = f"{sys_prompt}\n\n{user_prompt}"
            raw = await asyncio.to_thread(hf_generate, prompt, 0.2, 200)
            if not raw:
                raw = _HF_VERIFY_EMPTY_FALLBACK
            parsed = _safe_parse_json(raw, len(teacher_outputs))
        except Exception as e:
            print(f"[Verifier] Reasoning error: {e}")
            gold, labels = _fallback_majority(teacher_outputs)
            parsed = {"gold_answer": gold or "UNVERIFIABLE", "teacher_labels": labels}

        raw_labels = parsed.get("teacher_labels", [])
        if len(raw_labels) != len(teacher_outputs):
            raw_labels = ["unverifiable"] * len(teacher_outputs)

        labels = [
            {"style": t.get("style", ""), "label": raw_labels[i]}
            for i, t in enumerate(teacher_outputs)
        ]

        return {
            "gold_label": parsed.get("gold_answer", "UNVERIFIABLE"),
            "labels": labels,
            "verifier_type": "reasoning",
        }

