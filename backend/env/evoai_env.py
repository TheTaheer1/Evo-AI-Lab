"""
evoai_env.py
OpenEnv-compatible wrapper around EvoAIPipeline.
Exposes reset() / step() / state() / close() for server and evaluators.
"""

import os

try:
    from openenv import Environment
except ImportError:  # Compatibility fallback
    try:
        from openenv.core.env_server import Environment
    except ImportError:
        class Environment:  # pragma: no cover - local fallback
            pass

from backend.core.pipeline import EvoAIPipeline


class EvoAIEnv(Environment):
    def __init__(self, config: dict | None = None):
        config = config or {}
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.pipeline = EvoAIPipeline(groq_api_key=groq_api_key, config=config)
        self._state = {
            "step": 0,
            "calibration_map": self.pipeline.calibration_map.snapshot(),
            "reward_history": [],
            "last_step_result": {},
            "filter_pass_rate": 0.0,
            "filter_pass_rate_recent": 0.0,
            "zone_c_count": 0,
            "zone_b_count": 0,
            "green_count": 0,
            "difficulty_tier": self.pipeline.adversary.difficulty_tier,
            "total_moments": self.pipeline.dataset_builder.get_total_moments(),
        }

    def reset(self) -> dict:
        self.pipeline.reset()
        self.pipeline.dataset_builder.reset_dataset()
        snapshot = self.pipeline.calibration_map.snapshot()
        self._state = {
            "step": self.pipeline.step,
            "calibration_map": snapshot,
            "reward_history": self.pipeline.reward_history,
            "last_step_result": {},
            "filter_pass_rate": self.pipeline.disagreement_detector.pass_rate,
            "filter_pass_rate_recent": self.pipeline.filter_pass_rate_recent,
            "zone_c_count": snapshot.get("zone_c_count", 0),
            "zone_b_count": snapshot.get("zone_b_count", 0),
            "green_count": snapshot.get("green_count", 0),
            "difficulty_tier": self.pipeline.adversary.difficulty_tier,
            "total_moments": self.pipeline.dataset_builder.get_total_moments(),
        }
        return self._state

    async def step(self, action: dict | None = None) -> dict:
        step_result = await self.pipeline.run_step()
        snapshot = self.pipeline.calibration_map.snapshot()
        if step_result.get("skipped"):
            reward = None
        else:
            reward = float(step_result.get("reward", 0.0))

        self._state = {
            "step": self.pipeline.step,
            "calibration_map": snapshot,
            "reward_history": self.pipeline.reward_history,
            "last_step_result": step_result,
            "filter_pass_rate": self.pipeline.disagreement_detector.pass_rate,
            "filter_pass_rate_recent": self.pipeline.filter_pass_rate_recent,
            "zone_c_count": snapshot.get("zone_c_count", 0),
            "zone_b_count": snapshot.get("zone_b_count", 0),
            "green_count": snapshot.get("green_count", 0),
            "difficulty_tier": self.pipeline.adversary.difficulty_tier,
            "total_moments": self.pipeline.dataset_builder.get_total_moments(),
        }
        return {"observation": self._state, "reward": reward, "done": False, "info": step_result}

    def state(self) -> dict:
        snapshot = self.pipeline.calibration_map.snapshot()
        output = dict(self._state)
        output.update(
            {
                "step": self.pipeline.step,
                "calibration_map": snapshot,
                "reward_history": self.pipeline.reward_history,
                "filter_pass_rate": self.pipeline.disagreement_detector.pass_rate,
                "filter_pass_rate_recent": self.pipeline.filter_pass_rate_recent,
                "zone_c_count": snapshot.get("zone_c_count", 0),
                "zone_b_count": snapshot.get("zone_b_count", 0),
                "green_count": snapshot.get("green_count", 0),
                "difficulty_tier": self.pipeline.adversary.difficulty_tier,
                "total_moments": self.pipeline.dataset_builder.get_total_moments(),
            }
        )
        return output

    def close(self):
        self.pipeline.dataset_builder.flush_to_disk()

