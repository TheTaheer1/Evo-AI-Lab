"""
adversary.py
Generates questions targeting the student's weakest calibration nodes.
Zone C nodes (high confidence + wrong) are targeted first, then Zone B.

Difficulty tier is explicit (1-5) and adapts via increase/decrease_difficulty().
Starts at tier 2 (easy-moderate) regardless of what the map shows.
"""
import json
import re
import asyncio
import random
import httpx
from backend.core.calibration_map import CalibrationMap

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

_TIER_DESCRIPTIONS = {
    1: "EASY — single-concept, one-step problems. No tricks. Basic recall.",
    2: "MODERATE — single concept, requires applying it correctly once.",
    3: "HARD — multi-step; requires combining two concepts. One common misconception.",
    4: "EXPERT — edge cases, requires deep understanding; common intuition fails here.",
    5: "EXTREME — adversarial; designed to break confident but shallow reasoning.",
}

_TIER_NAMES = {1: "easy", 2: "moderate", 3: "hard", 4: "expert", 5: "extreme"}


class Adversary:
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.groq_api_key = groq_api_key
        self.model = model
        self.difficulty_tier: int = 2   # always start at tier 2 (spec requirement)
        self.force_escalate: bool = False
        self._followup_queue: list[str] = []

    # ── Difficulty controls ────────────────────────────────────────────────
    def increase_difficulty(self):
        """Bump difficulty tier up by 1 (max 5)."""
        old = self.difficulty_tier
        self.difficulty_tier = min(5, self.difficulty_tier + 1)
        print(f"[Adversary] Difficulty ↑ {old} → {self.difficulty_tier}")

    def decrease_difficulty(self):
        """Drop difficulty tier down by 1 (min 1)."""
        old = self.difficulty_tier
        self.difficulty_tier = max(1, self.difficulty_tier - 1)
        print(f"[Adversary] Difficulty ↓ {old} → {self.difficulty_tier}")

    def _difficulty_name(self) -> str:
        return _TIER_NAMES.get(self.difficulty_tier, "moderate")

    def reset_state(self):
        """Reset adaptive control state for a fresh training session."""
        self.difficulty_tier = 2
        self.force_escalate = False
        self._followup_queue = []

    def notify_node_graduated(self, node_key: str):
        """
        Called after a node transitions to Green so we do not keep targeting it
        in the follow-up queue (avoids one-step lag vs map refresh).
        """
        if node_key in self._followup_queue:
            self._followup_queue.remove(node_key)

    def _build_system_prompt(self, weak_nodes: list) -> str:
        tier_desc = _TIER_DESCRIPTIONS.get(self.difficulty_tier, _TIER_DESCRIPTIONS[2])
        diff_name = self._difficulty_name()
        node_lines = ""
        for node in weak_nodes[:5]:
            node_lines += (
                f"  - topic={node.get('topic')}, "
                f"type={node.get('question_type')}, "
                f"tier={node.get('difficulty_tier')}\n"
            )
        return (
            f"You are an adversarial question generator for AI calibration training.\n"
            f"Your goal: craft ONE question that exposes overconfidence in the student model.\n\n"
            f"DIFFICULTY TIER {self.difficulty_tier}/5: {tier_desc}\n\n"
            f"Weak areas to target (prioritise these):\n{node_lines}\n"
            f"Requirements:\n"
            f"1. Must have a single clear correct answer (not opinion-based).\n"
            f"2. Must plausibly fool an overconfident model at tier {self.difficulty_tier}.\n"
            f"3. Include a common misconception or counterintuitive element.\n"
            f"4. Return ONLY valid JSON, no preamble:\n"
            f'{{"question": "...", "topic": "math|code|factual|logic|planning", '
            f'"question_type": "math|code|factual|reasoning", '
            f'"difficulty_tier": "{diff_name}", '
            f'"target_node": "topic::question_type::{diff_name}"}}'
        )

    def _parse_response(self, raw: str, weak_nodes: list) -> dict:
        """Parse JSON from model response safely, with fallback."""
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: simple question at current tier
        fallback_qs = {
            1: "What is 15% of 200?",
            2: "Is the sum of two odd numbers always even? Explain briefly.",
            3: "If f(x) = x², what is f(f(2))?",
            4: "What does it mean for a set to be both open and closed in topology?",
            5: "Prove or disprove: every continuous function on [0,1] is uniformly continuous.",
        }
        return {
            "question":       fallback_qs.get(self.difficulty_tier, fallback_qs[2]),
            "topic":          weak_nodes[0].get("topic", "logic") if weak_nodes else "logic",
            "question_type":  "reasoning",
            "difficulty_tier": self._difficulty_name(),
            "target_node":    f"logic::reasoning::{self._difficulty_name()}",
        }

    def _refresh_followup_queue(self, calibration_map: CalibrationMap):
        """
        Keep a sticky queue of Zone B nodes with correct_streak >= 1.
        These nodes are one good answer away from Green and should be revisited first.
        """
        eligible = {
            node.key
            for node in calibration_map.nodes.values()
            if node.zone == "zone_b" and node.correct_streak >= 1
        }
        # Also include any zone_b node visited at least twice —
        # repeated exposure increases green conversion rate
        also_eligible = {
            node.key
            for node in calibration_map.nodes.values()
            if node.zone == "zone_b" and node.visit_count >= 2
        }
        eligible = eligible | also_eligible
        self._followup_queue = [k for k in self._followup_queue if k in eligible]
        for key in eligible:
            if key not in self._followup_queue:
                self._followup_queue.append(key)

    def _pick_weak_nodes(self, calibration_map: CalibrationMap) -> tuple[list[dict], str]:
        """
        Priority order (corrected):
          1) Zone C nodes  ← highest priority, most dangerous
          2) Follow-up Zone B nodes (correct_streak >= 1, near Green)
          3) Remaining Zone B nodes
          4) Novel fallback
        """
        self._refresh_followup_queue(calibration_map)

        zone_c_nodes = calibration_map.get_zone_c_nodes()
        if zone_c_nodes:
            random.shuffle(zone_c_nodes)
            selected = zone_c_nodes[:5]
            return ([
                {"topic": n.topic, "question_type": n.question_type, "difficulty_tier": n.difficulty_tier}
                for n in selected
            ], "Zone C")

        followup_nodes = [
            calibration_map.nodes[k]
            for k in self._followup_queue
            if k in calibration_map.nodes and calibration_map.nodes[k].zone == "zone_b"
        ]
        if followup_nodes:
            random.shuffle(followup_nodes)
            selected = followup_nodes[:5]
            return ([
                {"topic": n.topic, "question_type": n.question_type, "difficulty_tier": n.difficulty_tier}
                for n in selected
            ], "Zone B follow-up")

        zone_b_nodes = calibration_map.get_zone_b_nodes()
        if zone_b_nodes:
            random.shuffle(zone_b_nodes)
            selected = zone_b_nodes[:5]
            return ([
                {"topic": n.topic, "question_type": n.question_type, "difficulty_tier": n.difficulty_tier}
                for n in selected
            ], "Zone B")

        return ([{
            "topic": "logic",
            "question_type": "reasoning",
            "difficulty_tier": self._difficulty_name(),
        }], "novel")

    async def generate_question(self, calibration_map: CalibrationMap, step: int) -> dict:
        """
        Pull Zone C nodes first; fall back to Zone B; fall back to novel question.
        Returns a question dict.
        """
        weak_nodes, targeting = self._pick_weak_nodes(calibration_map)

        system_prompt = self._build_system_prompt(weak_nodes)
        user_prompt = (
            f"Generate ONE calibration question (tier {self.difficulty_tier}) "
            f"targeting {targeting} nodes. Return only the JSON object."
        )

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.85,
            "max_tokens": 512,
        }

        raw = ""
        try:
            retry_delays = (8, 16)
            exhausted_429 = False
            async with httpx.AsyncClient(timeout=30) as client:
                for attempt in range(3):
                    response = await client.post(_GROQ_URL, headers=headers, json=payload)
                    if response.status_code == 200:
                        raw = response.json()["choices"][0]["message"]["content"]
                        break
                    if response.status_code == 429:
                        if attempt < 2:
                            wait_seconds = retry_delays[attempt]
                            print(
                                f"[Adversary] API error 429 (attempt {attempt + 1}/3); "
                                f"retrying in {wait_seconds}s"
                            )
                            await asyncio.sleep(wait_seconds)
                            continue
                        exhausted_429 = True
                        break
                    print(f"[Adversary] API error {response.status_code}: {response.text[:200]}")
                    break
            if exhausted_429:
                topic = weak_nodes[0].get("topic", "logic") if weak_nodes else "logic"
                diff_name = self._difficulty_name()
                return {
                    "question": f"What is the definition of {topic}?",
                    "topic": topic,
                    "question_type": "factual",
                    "difficulty_tier": diff_name,
                    "target_node": f"{topic}::factual::{diff_name}",
                    "_difficulty_tier_num": self.difficulty_tier,
                }
        except Exception as e:
            msg = str(e).strip() or repr(e)
            print(f"[Adversary] API error ({type(e).__name__}): {msg}")

        result = self._parse_response(raw, weak_nodes)
        diff_name = self._difficulty_name()
        result.setdefault("topic", weak_nodes[0].get("topic", "logic") if weak_nodes else "logic")
        result.setdefault("question_type", "reasoning")
        result.setdefault("difficulty_tier", diff_name)
        result.setdefault("target_node",
                          f"{result['topic']}::{result['question_type']}::{diff_name}")
        result["_difficulty_tier_num"] = self.difficulty_tier

        # Follow-up mode: pin topic / type / tier to the near-Green node key so probe hits that node.
        if self._followup_queue:
            target_key = self._followup_queue[0]
            parts = target_key.split("::")
            if len(parts) == 3:
                result["topic"] = parts[0]
                result["question_type"] = parts[1]
                result["difficulty_tier"] = parts[2]
                result["target_node"] = target_key

        return result
