"""
calibration_probe.py
Query the student model for its answer and confidence on a question.
Plot the (confidence, correctness) pair on the calibration map.
"""
import re
import os
import asyncio
import httpx
from backend.core.calibration_map import CalibrationMap, ZONE_B

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

_STUDENT_SYSTEM = (
    "You are a knowledgeable assistant. Answer the question directly and concisely. "
    "At the very end of your response, on a new line, write exactly: "
    "CONFIDENCE: X  (where X is an integer from 0 to 10; "
    "0 = completely unsure, 10 = completely certain). "
    "IMPORTANT CALIBRATION RULES: "
    "- Use 9 or 10 ONLY if you could cite a textbook source verbatim. "
    "- Use 6, 7, or 8 for most answers where you are reasonably sure. "
    "- Use 3, 4, or 5 when you have partial knowledge or are estimating. "
    "- Use 0, 1, or 2 when you are mostly guessing. "
    "Overconfidence is heavily penalised. Be honest."
)

_COMPARE_SYSTEM = (
    "You are an answer comparison judge. "
    "Respond with a single word: YES if the student answer is semantically correct "
    "relative to the gold answer, NO if it is wrong or significantly incomplete."
)

# When the LLM judge is ambiguous, default to incorrect so the map and rewards keep updating (training-safe).
_AMBIGUOUS_AS_INCORRECT = os.environ.get(
    "EVOAI_PROBE_AMBIGUOUS_AS_INCORRECT", "true"
).strip().lower() in {"1", "true", "yes", "on"}

# Groq HTTP client: more patient connect/read; retries for flaky networks (ConnectError, timeouts).
try:
    _GROQ_MAX_RETRIES = max(1, int(os.environ.get("EVOAI_GROQ_MAX_RETRIES", "5")))
except ValueError:
    _GROQ_MAX_RETRIES = 5
_TRANSPORT_BACKOFF_SECS = (2, 4, 8, 16, 24)
_429_BACKOFF_SECS = (8, 16, 24)


def _groq_timeout() -> httpx.Timeout:
    return httpx.Timeout(60.0, connect=25.0, read=55.0, write=30.0)


def _log_probe_error(where: str, err: BaseException) -> None:
    """Groq/httpx often raises exceptions with empty str(e); log type + repr + HTTP details."""
    msg = str(err).strip() or repr(err)
    extra = ""
    if isinstance(err, httpx.HTTPStatusError) and err.response is not None:
        try:
            extra = f" status={err.response.status_code}"
            snippet = (err.response.text or "").replace("\n", " ")[:200]
            if snippet:
                extra += f" | {snippet!r}"
        except Exception:
            extra = f" status={err.response.status_code}"
    print(f"[CalibrationProbe] {where}{extra} ({type(err).__name__}): {msg}")


def _parse_yes_no_from_judge(raw: str) -> bool | None:
    """Extract YES/NO from judge output (handles 'YES.', 'Answer: NO', etc.)."""
    if not raw or not str(raw).strip():
        return None
    head = raw.strip().upper()
    # First alphanumeric token
    token_match = re.search(r"[A-Z]+", head)
    first_word = token_match.group(0) if token_match else head.split()[0] if head.split() else ""
    if first_word.startswith("Y") or head.startswith("YES") or "CORRECT" in head[:40]:
        return True
    if first_word.startswith("N") or head.startswith("NO") or "INCORRECT" in head[:40] or "WRONG" in head[:40]:
        return False
    return None


def _parse_confidence_int(text: str) -> int:
    """Extract CONFIDENCE integer (0–10) from student response."""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        match = re.search(r"CONFIDENCE:\s*(\d+)", line)
        if match:
            try:
                return max(0, min(10, int(match.group(1))))
            except ValueError:
                pass
    return 5  # Default: mid-confidence


def _strip_confidence_line(text: str) -> str:
    """Remove the CONFIDENCE line from the student's answer."""
    lines = text.strip().split("\n")
    body = [l for l in lines if not re.match(r"\s*CONFIDENCE:\s*\d+", l)]
    return "\n".join(body).strip()


class CalibrationProbe:
    def __init__(self, groq_api_key: str, student_model: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key
        self.student_model = student_model
        self._calibration_map: CalibrationMap | None = None

    def attach_map(self, calibration_map: CalibrationMap):
        """Attach the shared calibration map instance."""
        self._calibration_map = calibration_map

    async def _groq_post(
        self,
        messages: list,
        max_tokens: int = 400,
        temperature: float = 0.3,
        retries: int | None = None,
    ) -> str:
        """POST to Groq with 429 backoff and transport retries (ConnectError, read timeouts)."""
        if retries is None:
            retries = _GROQ_MAX_RETRIES
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        payload = {
            "model": self.student_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        timeout = _groq_timeout()
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    http2=False,
                ) as client:
                    resp = await client.post(_GROQ_URL, headers=headers, json=payload)
                if resp.status_code == 429:
                    if attempt < retries - 1:
                        wait_seconds = _429_BACKOFF_SECS[
                            min(attempt, len(_429_BACKOFF_SECS) - 1)
                        ]
                        print(
                            f"[CalibrationProbe] 429 in _groq_post attempt {attempt + 1}/{retries}; "
                            f"retrying in {wait_seconds}s"
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    print("[CalibrationProbe] 429 persisted in _groq_post; giving up")
                    return ""
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 429 and attempt < retries - 1:
                    wait_seconds = _429_BACKOFF_SECS[
                        min(attempt, len(_429_BACKOFF_SECS) - 1)
                    ]
                    print(
                        f"[CalibrationProbe] 429 in _groq_post attempt {attempt + 1}/{retries}; "
                        f"retrying in {wait_seconds}s"
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                _log_probe_error("_groq_post HTTPStatusError", e)
                return ""
            except httpx.RequestError as e:
                if attempt < retries - 1:
                    wait_seconds = _TRANSPORT_BACKOFF_SECS[
                        min(attempt, len(_TRANSPORT_BACKOFF_SECS) - 1)
                    ]
                    print(
                        f"[CalibrationProbe] transport error in _groq_post "
                        f"({type(e).__name__}) attempt {attempt + 1}/{retries}; "
                        f"retry in {wait_seconds}s"
                    )
                    await asyncio.sleep(wait_seconds)
                    continue
                _log_probe_error("_groq_post transport (gave up)", e)
                return ""
            except Exception as e:
                _log_probe_error("_groq_post unexpected", e)
                return ""
        return ""

    async def _is_correct(self, student_answer: str, correct_answer: str, question: str, confidence: int = 5) -> bool | None:
        """
        Semantic correctness via Groq YES/NO.
        Returns None when the API response is missing or ambiguous (do not treat as wrong).
        """
        if not correct_answer or correct_answer in ("UNVERIFIABLE", "MAJORITY_CORRECT"):
            return None
        if not student_answer.strip():
            return None

        messages = [
            {"role": "system", "content": _COMPARE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Gold answer: {correct_answer}\n"
                    f"Student answer: {student_answer}\n"
                    "Is the student answer correct? Reply YES or NO only."
                ),
            },
        ]
        raw = await self._groq_post(messages, max_tokens=64, temperature=0.0)
        verdict = _parse_yes_no_from_judge(raw) if raw else None
        if verdict is not None:
            return verdict

        # One retry with stricter instruction (reduces spurious Nones from terse models)
        clarify = (
            "You must answer with exactly one word: YES or NO. "
            "Is the student answer semantically correct relative to the gold answer?"
        )
        retry_messages = messages + [{"role": "user", "content": clarify}]
        raw2 = await self._groq_post(retry_messages, max_tokens=32, temperature=0.0)
        verdict = _parse_yes_no_from_judge(raw2) if raw2 else None
        if verdict is not None:
            return verdict

        if confidence >= 9:
            return False
        if _AMBIGUOUS_AS_INCORRECT:
            print(
                "[CalibrationProbe] Judge still ambiguous — marking incorrect "
                "(EVOAI_PROBE_AMBIGUOUS_AS_INCORRECT=true)"
            )
            return False
        return None

    async def probe(
        self,
        question: str,
        correct_answer: str,
        topic: str,
        question_type: str,
        difficulty_tier: str,
        calibration_map: CalibrationMap | None = None,
    ) -> dict:
        """
        Query the student model, parse confidence + answer, determine zone,
        update the calibration map, and return the probe result.
        """
        if calibration_map is None:
            calibration_map = self._calibration_map

        difficulty_tier = str(difficulty_tier)
        node_key = f"{topic}::{question_type}::{difficulty_tier}"
        prev_zone = ZONE_B   # default for brand-new nodes
        if calibration_map and node_key in calibration_map.nodes:
            prev_zone = calibration_map.nodes[node_key].zone

        # Query the student model (same transport retries + timeouts as judge calls)
        messages = [
            {"role": "system", "content": _STUDENT_SYSTEM},
            {"role": "user", "content": question},
        ]
        raw_response = await self._groq_post(
            messages, max_tokens=500, temperature=0.4, retries=_GROQ_MAX_RETRIES
        )

        if not (raw_response or "").strip():
            # Rate limit, transport failure, or empty body — do not mutate calibration map.
            return {
                "student_answer":  "",
                "confidence":      5,
                "is_correct":      None,
                "correctness_unknown": True,
                "zone":            prev_zone,
                "prev_zone":       prev_zone,
                "node_key":        node_key,
                "topic":           topic,
                "question_type":   question_type,
                "difficulty_tier": str(difficulty_tier),
            }

        confidence = _parse_confidence_int(raw_response)
        student_answer = _strip_confidence_line(raw_response)

        # Semantic correctness check
        is_correct = await self._is_correct(student_answer, correct_answer, question, confidence=confidence)

        # Update the calibration map only when correctness is known.
        new_zone = prev_zone
        if calibration_map and is_correct is not None:
            new_zone = calibration_map.update_node(
                topic=topic,
                question_type=question_type,
                difficulty_tier=difficulty_tier,
                is_correct=is_correct,
                confidence=float(confidence),
            )

        return {
            "student_answer":  student_answer,
            "confidence":      confidence,
            "is_correct":      is_correct,
            "correctness_unknown": is_correct is None,
            "zone":            new_zone,
            "prev_zone":       prev_zone,
            "node_key":        node_key,
            "topic":           topic,
            "question_type":   question_type,
            "difficulty_tier": difficulty_tier,
        }
