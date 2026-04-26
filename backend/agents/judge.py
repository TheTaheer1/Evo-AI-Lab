"""
judge.py
Synthesise teacher outputs, verifier labels, probe result, and critic scores
into one gold answer (training positive) and one failure record (contrastive negative).
"""
import asyncio
import os
import httpx

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

_CORRECTION_SYSTEM = (
    "You are a calibration training expert. "
    "Given a question, a wrong answer, and the correct gold answer, "
    "write 2–3 sentences that explain EXACTLY why the wrong answer fails and "
    "what the correct reasoning process should be. Be specific and educational. "
    "Do not repeat the answers verbatim — explain the conceptual error."
)


class Judge:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key
        self.model = model
        self.low_tpm_mode = os.environ.get("EVOAI_LOW_TPM_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

    async def _groq_call(self, messages: list, max_tokens: int = 300) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        try:
            retry_delays = (8, 16)
            async with httpx.AsyncClient(timeout=30) as client:
                for attempt in range(3):
                    resp = await client.post(_GROQ_URL, headers=headers, json=payload)
                    if resp.status_code == 429:
                        if attempt < 2:
                            wait_seconds = retry_delays[attempt]
                            print(
                                f"[Judge] API error 429 (attempt {attempt + 1}/3); "
                                f"retrying in {wait_seconds}s"
                            )
                            await asyncio.sleep(wait_seconds)
                            continue
                        print("[Judge] API error 429 persisted after 3 attempts; using local fallback")
                        return ""
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Judge] API error: {e}")
            return ""

    async def synthesise(
        self,
        question: str,
        teacher_outputs: list,
        verified_labels: list,
        probe_result: dict,
        critic_scores: list,
    ) -> dict:
        """
        Synthesise all inputs into a gold_answer and failure_answer pair.

        Args:
            question:         The question string.
            teacher_outputs:  List of teacher dicts (style, answer, reasoning, confidence).
            verified_labels:  List of {style, label} dicts from verifier.
            probe_result:     Output from CalibrationProbe.probe().
            critic_scores:    List of augmented teacher dicts from Critic.evaluate_all().

        Returns:
            {
                "gold_answer": str,
                "failure_answer": str,
                "failure_reason": str,
                "correction": str,
                "is_valid_pair": bool,
                "question": str,
                "topic": str,
                "question_type": str,
                "difficulty_tier": str,
                "clearly_answerable": bool,
                "hallucination_detected": bool,
            }
        """
        # Build label lookup: style → label
        label_map = {v.get("style", ""): v.get("label", "unverifiable") for v in verified_labels}

        # Build reasoning_score lookup: style → reasoning_score
        score_map = {c.get("style", ""): c.get("reasoning_score", 0.0) for c in critic_scores}

        # Filter correct teachers
        correct_teachers = [
            t for t in teacher_outputs
            if label_map.get(t.get("style", ""), "unverifiable") == "correct"
        ]

        # Filter incorrect teachers
        incorrect_teachers = [
            t for t in teacher_outputs
            if label_map.get(t.get("style", ""), "unverifiable") == "incorrect"
        ]

        # ── Gold answer ────────────────────────────────────────────────────
        gold_answer = "UNVERIFIABLE"
        if correct_teachers:
            # Pick the correct teacher with the highest reasoning_score
            best_correct = max(
                correct_teachers,
                key=lambda t: score_map.get(t.get("style", ""), 0.0),
            )
            gold_answer = best_correct.get("answer", "UNVERIFIABLE")

        # ── Failure answer (contrastive negative) ──────────────────────────
        failure_answer = ""
        failure_reason = ""
        correction = ""
        is_valid_pair = False

        if incorrect_teachers and gold_answer != "UNVERIFIABLE":
            # Pick the most confidently wrong incorrect teacher
            most_confident_wrong = max(
                incorrect_teachers,
                key=lambda t: float(t.get("confidence", 0.0)),
            )
            failure_answer = most_confident_wrong.get("answer", "")

            stu = (probe_result.get("student_answer") or "").strip()
            if self.low_tpm_mode:
                failure_reason = "Most confident wrong answer selected for contrastive learning."
                correction = (
                    f"Correct answer: {gold_answer}. "
                    "The selected failure answer is incorrect and should be replaced by the gold reasoning."
                )
                if stu:
                    correction += f" Student said: {stu[:280]}{'…' if len(stu) > 280 else ''}"
            else:
                # Generate failure reason and correction in parallel
                failure_reason, correction = await asyncio.gather(
                    self._generate_failure_reason(question, failure_answer),
                    self._generate_correction(
                        question, failure_answer, gold_answer, student_answer=stu
                    ),
                )
            is_valid_pair = True

        # Hallucination detection: only when probe explicitly says wrong (not None/unknown)
        hallucination_detected = (
            probe_result.get("is_correct") is False
            and probe_result.get("confidence", 0) >= 8
        )

        return {
            "gold_answer": gold_answer,
            "failure_answer": failure_answer,
            "failure_reason": failure_reason,
            "correction": correction,
            "is_valid_pair": is_valid_pair,
            "question": question,
            "topic": probe_result.get("topic", ""),
            "question_type": probe_result.get("question_type", ""),
            "difficulty_tier": probe_result.get("difficulty_tier", ""),
            "clearly_answerable": gold_answer != "UNVERIFIABLE",
            "hallucination_detected": hallucination_detected,
        }

    async def _generate_failure_reason(self, question: str, wrong_answer: str) -> str:
        """Generate a brief reason why this answer is wrong."""
        messages = [
            {"role": "system", "content": "You are an expert evaluator. In 1–2 sentences, state why the following answer is incorrect or flawed for the given question. Be direct and specific."},
            {"role": "user", "content": f"Question: {question}\n\nWrong answer: {wrong_answer}"},
        ]
        result = await self._groq_call(messages, max_tokens=150)
        return result if result else "The answer contains factual or reasoning errors."

    async def _generate_correction(
        self,
        question: str,
        wrong_answer: str,
        gold_answer: str,
        student_answer: str = "",
    ) -> str:
        """
        Call Groq to produce a 2–3 sentence explanation of exactly why
        the wrong answer fails and what the correct reasoning is.
        """
        stu_block = (
            f"\n\nStudent model answer (for calibration context): {student_answer}\n"
            if (student_answer or "").strip()
            else ""
        )
        messages = [
            {"role": "system", "content": _CORRECTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Wrong contrastive answer: {wrong_answer}\n\n"
                    f"Gold answer: {gold_answer}"
                    f"{stu_block}\n\n"
                    "Explain the conceptual error in 2–3 sentences. "
                    "If a student answer is given, briefly relate why it is or is not aligned with the gold."
                ),
            },
        ]
        result = await self._groq_call(messages, max_tokens=200)
        return result if result else (
            f"The correct answer is: {gold_answer}. "
            "The wrong answer misidentifies the key concept or applies incorrect reasoning."
        )
