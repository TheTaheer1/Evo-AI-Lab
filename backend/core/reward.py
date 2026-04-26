"""
reward.py
Dual-axis reward function: accuracy + calibration + reasoning.
Features:
 - Warmup phase (steps 0-14): penalties scaled by 0.5 so early training isn't crushed
 - Zone C penalty reduced to −0.25 (was −0.40) — less aggressive early on
 - Zone shift fires when node zone IMPROVED vs start-of-step zone (prev_zone param)
 - Per-step breakdown log for full debugging visibility
"""
from backend.core.calibration_map import ZONE_C, ZONE_B, ZONE_GREEN  # noqa: F401

_WARMUP_STEPS = 15      # steps 0..14 get 0.5x penalty multiplier


class RewardCalculator:
    # Accuracy axis
    CORRECT_ANSWER         = +0.35
    WRONG_HIGH_CONFIDENCE  = -0.25   # reduced from -0.40
    CORRECT_LOW_CONFIDENCE = -0.10   # slight penalty for being right but unsure

    # Calibration axis
    ZONE_C_TO_B_SHIFT      = +0.40
    ZONE_B_TO_GREEN_SHIFT  = +0.50
    APPROPRIATE_UNCERTAINTY = +0.15  # wrong but low confidence (model knew it)

    # Reasoning axis
    HIGH_REASONING         = +0.20

    # Penalty axis
    OVER_REFUSAL           = -0.20
    HALLUCINATION          = -0.40   # reduced from -0.50

    def compute(
        self,
        probe_result: dict,
        judgment: dict,
        critic_scores: list,
        training_step: int = 100,   # step index; default > warmup so no discount applied
    ) -> dict:
        """
        Compute total reward and per-axis breakdown.

        Args:
            probe_result:  output from CalibrationProbe.probe() — includes
                           is_correct (True/False/None), confidence (0-10), zone, prev_zone
            judgment:      output from Judge.synthesise()
            critic_scores: list of dicts with reasoning_score per teacher
            training_step: current global step (for warmup scaling)

        Returns:
            {total: float, breakdown: dict, is_positive: bool}
        """
        total = 0.0
        breakdown: dict = {}

        # True / False / None — None means compare failed; do not treat as incorrect.
        is_correct   = probe_result.get("is_correct")
        confidence   = probe_result.get("confidence", 5)   # 0–10 scale
        zone         = probe_result.get("zone", ZONE_B)
        # prev_zone = node zone BEFORE this step answered (set by CalibrationProbe)
        prev_zone    = probe_result.get("prev_zone", zone)

        # Warmup multiplier — soften penalties in early training
        in_warmup = training_step < _WARMUP_STEPS
        penalty_scale = 0.5 if in_warmup else 1.0

        # ── Accuracy axis ──────────────────────────────────────────────────
        if is_correct is True:
            total += self.CORRECT_ANSWER
            breakdown["correct_answer"] = self.CORRECT_ANSWER

        if is_correct is False and confidence >= 7:
            penalty = self.WRONG_HIGH_CONFIDENCE * penalty_scale
            total += penalty
            breakdown["wrong_high_confidence"] = round(penalty, 4)

        if is_correct is True and confidence < 4:
            total += self.CORRECT_LOW_CONFIDENCE
            breakdown["correct_low_confidence"] = self.CORRECT_LOW_CONFIDENCE

        # ── Calibration axis ───────────────────────────────────────────────
        # Zone improvement: node was in Zone C at START of step, now improved
        if prev_zone == ZONE_C and zone in (ZONE_B, ZONE_GREEN):
            total += self.ZONE_C_TO_B_SHIFT
            breakdown["zone_shift_c_to_b"] = self.ZONE_C_TO_B_SHIFT

        # Also reward Zone B → Green improvement — higher reward than C→B
        if prev_zone == ZONE_B and zone == ZONE_GREEN:
            total += self.ZONE_B_TO_GREEN_SHIFT
            breakdown["zone_shift_b_to_green"] = self.ZONE_B_TO_GREEN_SHIFT

        # Appropriate uncertainty: model was wrong but showed some self-awareness (conf <= 6)
        if is_correct is False and confidence <= 6:
            scale = 1.0 if confidence <= 4 else 0.5   # partial credit for conf 5-6
            reward_val = round(self.APPROPRIATE_UNCERTAINTY * scale, 4)
            total += reward_val
            breakdown["appropriate_uncertainty"] = reward_val

        # ── Reasoning axis ─────────────────────────────────────────────────
        if critic_scores:
            avg_reasoning = sum(
                c.get("reasoning_score", 0.0) for c in critic_scores
            ) / len(critic_scores)
            if avg_reasoning > 8.0:
                total += self.HIGH_REASONING
                breakdown["high_reasoning"] = self.HIGH_REASONING
        else:
            avg_reasoning = 0.0

        # ── Hallucination ──────────────────────────────────────────────────
        if judgment.get("hallucination_detected", False):
            if "wrong_high_confidence" not in breakdown:
                penalty = self.HALLUCINATION * penalty_scale
                total += penalty
                breakdown["hallucination"] = round(penalty, 4)
            else:
                breakdown["hallucination_skipped"] = "merged_into_wrong_high_confidence"

        # ── Over-refusal ───────────────────────────────────────────────────
        student_answer = probe_result.get("student_answer", "").strip().lower()
        refusal_phrases = ("i cannot", "i'm unable", "i am unable")
        if (
            student_answer.startswith(refusal_phrases)
            and judgment.get("is_valid_pair", False)
            and judgment.get("clearly_answerable", False)
        ):
            penalty = self.OVER_REFUSAL * penalty_scale
            total += penalty
            breakdown["over_refusal"] = round(penalty, 4)

        total = round(total, 4)

        # ── Breakdown summary log (always returned) ────────────────────────
        breakdown["_total"]          = total
        breakdown["_in_warmup"]      = in_warmup
        breakdown["_penalty_scale"]  = penalty_scale
        breakdown["_avg_reasoning"]  = round(avg_reasoning, 2)

        print(
            f"[Reward] step={training_step} total={total:+.3f} "
            f"correct={is_correct!r} conf={confidence} "
            f"zone={prev_zone}→{zone} warmup={in_warmup} | {breakdown}"
        )

        return {
            "total":       total,
            "breakdown":   breakdown,
            "is_positive": total > 0,
        }
