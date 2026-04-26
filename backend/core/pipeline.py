"""
pipeline.py
Master orchestrator for the EvoAI Lab training loop.
Runs: adversary → teachers → disagreement filter → verifier →
calibration probe → critic → judge → reward → dataset → loop.

Adaptive difficulty (after each real step):
  Uses the last N disagreement outcomes (pass=1 / skip=0), N = _ADAPT_WINDOW.
  If that rolling pass rate is < 0.30 for N consecutive real steps → decrease_difficulty()
  If it is > 0.60 for N consecutive real steps → increase_difficulty()
"""
import asyncio
import logging
import os
from collections import deque

from backend.agents.adversary import Adversary
from backend.agents.teacher import TeacherPanel
from backend.agents.calibration_probe import CalibrationProbe
from backend.agents.critic import Critic
from backend.agents.judge import Judge
from backend.core.disagreement import DisagreementDetector
from backend.core.verifier import Verifier
from backend.core.calibration_map import CalibrationMap, ZONE_GREEN
from backend.core.reward import RewardCalculator
from backend.core.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

# Adaptive difficulty thresholds
_PASS_RATE_TOO_LOW  = 0.30   # < 30% passing → decrease difficulty
_PASS_RATE_TOO_HIGH = 0.60   # > 60% passing → increase difficulty
_ADAPT_WINDOW       = 5      # consecutive real steps before adapting

# Delay between sequential LLM-heavy stages (0 = fastest; higher reduces TPM bursts)
_PIPELINE_STAGE_SLEEP_SECS = float(os.environ.get("EVOAI_PIPELINE_STAGE_SLEEP_SECS", "2.0"))


class EvoAIPipeline:
    def __init__(self, groq_api_key: str, config: dict = None):
        if config is None:
            config = {}

        self.groq_api_key = groq_api_key
        self.config = config

        # ── Agents ─────────────────────────────────────────────────────────
        self.adversary       = Adversary(groq_api_key)      # starts at tier 2
        self.teacher_panel   = TeacherPanel(groq_api_key)
        self.calibration_probe = CalibrationProbe(groq_api_key)
        self.critic          = Critic(groq_api_key)
        self.judge           = Judge(groq_api_key)

        # ── Core ───────────────────────────────────────────────────────────
        self.disagreement_detector = DisagreementDetector(
            threshold=config.get("disagreement_threshold", 0.70)
        )
        self.verifier = Verifier(
            groq_api_key,
            faiss_index_path=config.get("faiss_index_path", "eval/faiss_index"),
        )
        self.calibration_map     = CalibrationMap()
        self.reward_calculator   = RewardCalculator()
        self.dataset_builder     = DatasetBuilder(
            output_dir=config.get("output_dir", "data/"),
            eval_path=config.get("eval_path", "eval/held_out_eval.json"),
        )

        # Attach calibration map to probe so it can look up prev_zone
        self.calibration_probe.attach_map(self.calibration_map)

        # ── State ──────────────────────────────────────────────────────────
        self.step = 0
        self.real_step = 0                  # non-skipped steps only
        self.reward_history: list = []
        self.filter_pass_rate_history: list = []

        # Adaptive difficulty tracking
        self._low_pass_streak  = 0          # consecutive real steps with low pass rate
        self._high_pass_streak = 0          # consecutive real steps with high pass rate
        self._disagreement_recent: deque[int] = deque(maxlen=_ADAPT_WINDOW)

    @property
    def filter_pass_rate_recent(self) -> float:
        """Mean disagreement-filter pass rate over the last up to N attempts (N = _ADAPT_WINDOW)."""
        if not self._disagreement_recent:
            return 0.0
        return sum(self._disagreement_recent) / len(self._disagreement_recent)

    async def run_step(self) -> dict:
        """Execute one full training step. Returns a result dict."""
        # Step 1: Adversary generates a targeted question
        question_data = await self.adversary.generate_question(
            self.calibration_map, self.step
        )
        await asyncio.sleep(_PIPELINE_STAGE_SLEEP_SECS)

        # Step 2: Three teachers answer in parallel
        teacher_outputs = await self.teacher_panel.answer_all(question_data["question"])
        await asyncio.sleep(_PIPELINE_STAGE_SLEEP_SECS)

        # Step 3: Disagreement filter
        passed = self.disagreement_detector.filter(teacher_outputs)
        pass_rate = self.disagreement_detector.pass_rate
        self._disagreement_recent.append(1 if passed else 0)
        rolling_pass_rate = self.filter_pass_rate_recent

        if not passed:
            logger.info(
                f"[Step {self.step}] Skipped — teachers agree "
                f"(lifetime_pass={pass_rate:.2f} recent_pass={rolling_pass_rate:.2f})"
            )
            self.step += 1
            return {
                "skipped":    True,
                "reason":     "teachers_agree",
                "step":       self.step - 1,
                "pass_rate":  pass_rate,
                "pass_rate_recent": rolling_pass_rate,
                "difficulty_tier": self.adversary.difficulty_tier,
            }

        # ── Real step from here ────────────────────────────────────────────

        # Step 4: Verifier labels each teacher's answer
        verified = await self.verifier.verify_all(question_data, teacher_outputs)
        await asyncio.sleep(_PIPELINE_STAGE_SLEEP_SECS)

        # Step 5: Calibration probe — captures prev_zone BEFORE update, then updates map
        probe_result = await self.calibration_probe.probe(
            question=question_data["question"],
            correct_answer=verified["gold_label"],
            topic=question_data.get("topic", "logic"),
            question_type=question_data.get("question_type", "reasoning"),
            difficulty_tier=str(question_data.get("difficulty_tier", "moderate")),
            calibration_map=self.calibration_map,
        )
        await asyncio.sleep(_PIPELINE_STAGE_SLEEP_SECS)

        if (
            probe_result.get("zone") == ZONE_GREEN
            and probe_result.get("prev_zone") != ZONE_GREEN
        ):
            nk = probe_result.get("node_key")
            if nk:
                self.adversary.notify_node_graduated(nk)

        # Step 6a: Critic evaluates reasoning quality
        critic_scores = await self.critic.evaluate_all(
            question_data["question"], teacher_outputs
        )
        await asyncio.sleep(_PIPELINE_STAGE_SLEEP_SECS)

        # Step 6b: Judge synthesises gold answer + failure correction
        judgment = await self.judge.synthesise(
            question=question_data["question"],
            teacher_outputs=teacher_outputs,
            verified_labels=verified["labels"],
            probe_result=probe_result,
            critic_scores=critic_scores,
        )
        await asyncio.sleep(_PIPELINE_STAGE_SLEEP_SECS)

        # Step 7: Compute reward (with training_step for warmup scaling)
        reward = self.reward_calculator.compute(
            probe_result,
            judgment,
            critic_scores,
            training_step=self.real_step,
        )

        # Step 8: Persist training pair — also save Zone B failures
        zone = probe_result.get("zone", "zone_b")
        is_valid = judgment.get("is_valid_pair", False)
        ic = probe_result.get("is_correct")
        # Unknown correctness (API failure): do not log as failure or poison the dataset.
        is_learning_moment = is_valid and (
            ic is False
            or (ic is True and zone in ("zone_c", "zone_b"))
        )
        if is_learning_moment:
            self.dataset_builder.add_training_pair(
                judgment, reward,
                zone=zone,
                confidence=probe_result.get("confidence", 5),
                is_correct=bool(ic),
                student_answer=probe_result.get("student_answer") or "",
            )

        # Log to failure panel even when no valid pair was formed
        failure_only_logged = False
        if (
            probe_result.get("is_correct") is False
            or (probe_result.get("is_correct") is None
                and probe_result.get("zone") in ("zone_c", "zone_b")
                and probe_result.get("confidence", 5) >= 7)
        ) and not judgment.get("is_valid_pair", False):
            self.dataset_builder.add_failure_only(
                question=question_data.get("question", ""),
                topic=probe_result.get("topic", ""),
                question_type=probe_result.get("question_type", ""),
                difficulty_tier=probe_result.get("difficulty_tier", ""),
                student_answer=probe_result.get("student_answer", ""),
                correction=judgment.get("correction")
                or f"The model answered incorrectly with high confidence ({probe_result.get('confidence', 5)}/10) on a {probe_result.get('difficulty_tier','?')} {probe_result.get('topic','?')} question.",
                zone=probe_result.get("zone", "zone_b"),
                confidence=float(probe_result.get("confidence", 5)),
            )
            failure_only_logged = True

        # Track reward history
        self.reward_history.append({
            "step":        self.step,
            "reward":      reward["total"],
            "is_positive": reward["is_positive"],
        })

        # ── Adaptive difficulty based on rolling disagreement pass rate ────
        self.real_step += 1
        if rolling_pass_rate < _PASS_RATE_TOO_LOW:
            self._low_pass_streak  += 1
            self._high_pass_streak  = 0
        elif rolling_pass_rate > _PASS_RATE_TOO_HIGH:
            self._high_pass_streak += 1
            self._low_pass_streak   = 0
        else:
            self._low_pass_streak   = 0
            self._high_pass_streak  = 0

        if self._low_pass_streak >= _ADAPT_WINDOW:
            self.adversary.decrease_difficulty()
            self._low_pass_streak = 0

        if self._high_pass_streak >= _ADAPT_WINDOW:
            self.adversary.increase_difficulty()
            self._high_pass_streak = 0

        # If the map is still mostly non-green, do not sit at extreme difficulty
        zc = self.calibration_map.get_zone_counts()
        _n_nodes = max(zc["zone_c"] + zc["zone_b"] + zc["green"], 1)
        if zc["green"] / _n_nodes < 0.30 and self.adversary.difficulty_tier >= 5:
            self.adversary.difficulty_tier = 4
            logger.info("[Pipeline] Green fraction < 0.30 — capping difficulty at tier 4.")

        # Flush dataset every 50 real steps
        if self.real_step > 0 and self.real_step % 50 == 0:
            self.dataset_builder.flush_to_disk()
            logger.info(f"[Step {self.step}] Dataset flushed to disk.")

        self.step += 1
        map_snapshot = self.calibration_map.snapshot()

        # WebSocket / UI: mirror the exact row appended to failure_log (unique step, augmented correction)
        failure_feed = None
        if (is_learning_moment or failure_only_logged) and self.dataset_builder.failure_log:
            failure_feed = dict(self.dataset_builder.failure_log[-1])

        return {
            "skipped":          False,
            "step":             self.step - 1,
            "question":         question_data["question"],
            "question_data":    question_data,
            "teacher_outputs":  teacher_outputs,
            "verified":         verified,
            "probe_result":     probe_result,
            "critic_scores":    critic_scores,
            "judgment":         judgment,
            "reward":           reward["total"],
            "reward_breakdown": reward["breakdown"],
            "is_positive":      reward["is_positive"],
            "calibration_map":  map_snapshot,
            "zone_counts": {
                "zone_c": map_snapshot.get("zone_c_count", 0),
                "zone_b": map_snapshot.get("zone_b_count", 0),
                "green":  map_snapshot.get("green_count", 0),
            },
            "pass_rate":         pass_rate,
            "pass_rate_recent":  rolling_pass_rate,
            "difficulty_tier":   self.adversary.difficulty_tier,
            "failure":           failure_feed,
        }

    async def run_loop(self, max_steps: int = 1000):
        """Run the training loop up to max_steps steps."""
        logger.info(f"[Pipeline] Starting training loop for {max_steps} steps.")
        for _ in range(max_steps):
            try:
                result = await self.run_step()
                if result.get("skipped"):
                    logger.info(f"  → skipped (reason: {result.get('reason')})")
                else:
                    logger.info(
                        f"  → step={result['step']} "
                        f"reward={result['reward']:.3f} "
                        f"zone={result['probe_result'].get('zone','?')} "
                        f"tier={result['difficulty_tier']}"
                    )
            except Exception as e:
                logger.error(f"[Pipeline] Step error: {e}", exc_info=True)

        self.dataset_builder.flush_to_disk()
        logger.info("[Pipeline] Training loop complete. Dataset flushed.")

    def reset(self):
        """Reset step counter, calibration map, reward history, and difficulty."""
        self.step             = 0
        self.real_step        = 0
        self.reward_history   = []
        self.filter_pass_rate_history = []
        self._low_pass_streak  = 0
        self._high_pass_streak = 0
        self._disagreement_recent.clear()
        self.calibration_map.reset()
        self.adversary.reset_state()
        logger.info("[Pipeline] Reset complete.")
