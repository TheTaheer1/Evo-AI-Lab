"""
dataset_builder.py
Store training pairs (gold answer + failure correction) and maintain the
held-out eval set. Writes JSONL for GRPO fine-tuning.

Learning moments = any step where the student was WRONG (Zone C or Zone B),
not just Zone C. Zone B uncertain+wrong is also a training signal.
"""
import json
import os
from pathlib import Path


def _gold_displayable(gold: str) -> bool:
    if not gold or not str(gold).strip():
        return False
    u = str(gold).strip().upper()
    return u not in {"UNVERIFIABLE", "NONE", "NULL", "MAJORITY_CORRECT"}


def _augment_correction(correction: str, gold_hint: str, student_answer: str) -> str:
    """Ensure the learning panel always has readable text when possible."""
    c = (correction or "").strip()
    if c:
        return c
    if _gold_displayable(gold_hint):
        return (
            f"Verified correct answer: {gold_hint.strip()}. "
            "Prefer this reasoning over the mistaken line shown above."
        )
    if (student_answer or "").strip():
        return (
            "The student response did not match the verified answer for this skill. "
            "Keep training — the map will update as calibration improves."
        )
    return (
        "No contrastive explanation was generated (API or verifier gap). "
        "Check Groq connectivity and run another step."
    )


class DatasetBuilder:
    def __init__(
        self,
        output_dir: str = "data/",
        eval_path: str = "eval/held_out_eval.json",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_path = eval_path

        self.training_pairs: list = []
        self.reward_log: list = []
        self.failure_log: list = []    # all learning moments (wrong answers)
        self.step_counter: int = 0
        self.total_moments: int = 0    # cumulative count, never resets

        # Load eval set if it exists
        self.eval_set: list = []
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    self.eval_set = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[DatasetBuilder] Warning: could not load eval set: {e}")

    def add_training_pair(
        self,
        judgment: dict,
        reward: dict,
        zone: str = "zone_b",
        confidence: float = 5.0,
        is_correct: bool = False,
        student_answer: str = "",
    ):
        """
        Append one (chosen, rejected) DPO pair with reward metadata.
        Also adds to failure_log for the What the Model Learned panel.
        """
        pair = {
            "prompt":          judgment.get("question", ""),
            "chosen":          judgment.get("gold_answer", ""),
            "rejected":        judgment.get("failure_answer", ""),
            "reward":          reward.get("total", 0.0),
            "breakdown":       reward.get("breakdown", {}),
            "step":            self.step_counter,
            "topic":           judgment.get("topic", ""),
            "question_type":   judgment.get("question_type", ""),
            "difficulty_tier": judgment.get("difficulty_tier", ""),
            "correction":      judgment.get("correction", ""),
            "zone":            zone,
            "confidence":      confidence,
            "is_correct":      is_correct,
            "learned":         False,  # updated later if revisit is correct
        }
        self.training_pairs.append(pair)
        self.reward_log.append({
            "step":        self.step_counter,
            "reward":      reward.get("total", 0.0),
            "is_positive": reward.get("is_positive", False),
        })

        # Learning panel: student_answer = what the student model said; failure_answer = DPO rejected (often a wrong teacher).
        sa = (student_answer or "").strip()
        gh = judgment.get("gold_answer", "")
        corr_raw = judgment.get("correction", "")
        failure_entry = {
            "topic":           judgment.get("topic", ""),
            "question_type":   judgment.get("question_type", ""),
            "difficulty_tier": judgment.get("difficulty_tier", ""),
            "question":        (judgment.get("question") or "")[:400],
            "student_answer":  sa,
            "failure_answer":  judgment.get("failure_answer", ""),
            "correction":      _augment_correction(corr_raw, gh, sa),
            "gold_hint":       gh if _gold_displayable(gh) else "",
            "zone":            zone,
            "confidence":      confidence,
            "is_correct":      is_correct,
            "step":            self.step_counter,
            "learned":         bool(is_correct),
        }
        self.failure_log.append(failure_entry)

        # Retroactively mark earlier same-topic+type failures as learned
        # when the student just answered the same topic+type correctly
        if is_correct:
            topic = judgment.get("topic", "")
            qtype = judgment.get("question_type", "")
            dtier = (judgment.get("difficulty_tier") or "").strip()
            for entry in self.failure_log[:-1]:  # all but the entry just appended
                if (
                    entry.get("topic") == topic
                    and entry.get("question_type") == qtype
                    and (entry.get("difficulty_tier") or "").strip() == dtier
                    and not entry.get("learned", False)
                ):
                    entry["learned"] = True

        self.total_moments += 1
        self.step_counter += 1

    def add_failure_only(
        self,
        question: str,
        topic: str,
        question_type: str,
        difficulty_tier: str,
        student_answer: str,
        correction: str,
        zone: str,
        confidence: float,
    ):
        """
        Log a learning moment even when no valid chosen/rejected pair exists.
        Uses the same step_counter as add_training_pair so every moment has a unique id.
        """
        sa = (student_answer or "").strip()
        entry = {
            "topic":           topic,
            "question_type":   question_type,
            "difficulty_tier": difficulty_tier,
            "question":        (question or "")[:400],
            "student_answer":  sa,
            "failure_answer":  sa,
            "correction":      _augment_correction(correction, "", sa),
            "gold_hint":       "",
            "zone":            zone,
            "confidence":      confidence,
            "is_correct":      False,
            "step":            self.step_counter,
            "learned":         False,
        }
        self.failure_log.append(entry)
        self.total_moments += 1
        self.step_counter += 1

    def reset_dataset(self):
        pairs_path = self.output_dir / "training_pairs.jsonl"

        # Clear file
        with open(pairs_path, "w", encoding="utf-8") as f:
            pass

        # Clear in-memory buffer
        self.training_pairs = []

        print("[DATASET RESET] training_pairs.jsonl cleared")

    def flush_to_disk(self):
        """Append training pairs to data/training_pairs.jsonl."""
        pairs_path = self.output_dir / "training_pairs.jsonl"
        with open(pairs_path, "a", encoding="utf-8") as f:
            for pair in self.training_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        reward_path = self.output_dir / "reward_log.json"
        with open(reward_path, "w", encoding="utf-8") as f:
            json.dump(self.reward_log, f, indent=2, ensure_ascii=False)

        # Clear only pending pair buffer after successful append to avoid duplicates.
        self.training_pairs = []

    def get_reward_curve(self) -> list:
        """Return the full reward log."""
        return self.reward_log

    def get_recent_failures(self, n: int = 10) -> list:
        """Return last n learning moments (all wrong answers, Zone C or B)."""
        return list(reversed(self.failure_log[-n:]))

    def get_total_moments(self) -> int:
        """Total accumulated learning moments."""
        return self.total_moments
