"""
calibration_map.py
Topic graph where each node is a (topic, question_type, difficulty_tier) triple.
Nodes are coloured by zone. The adversary reads this; the frontend visualises it.

Zone transition rules
─────────────────────
Zone C → Zone B   : student answers correctly (any confidence)
Zone B → Green    : two consecutive correct answers with conf ≥ 6, OR one correct with conf ≥ 8
Green  → Zone B   : student answers wrong once (buffered regression)
Zone B → Zone C   : two consecutive wrong answers while in Zone B
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict

ZONE_C = "zone_c"       # high confidence + wrong  (confidence >= 7, incorrect)
ZONE_B = "zone_b"       # uncertain + wrong         (confidence < 7, incorrect)
ZONE_GREEN = "green"    # calibrated — correct


@dataclass
class CalibrationNode:
    topic: str
    question_type: str
    difficulty_tier: str
    zone: str = ZONE_B
    confidence_history: List[float] = field(default_factory=list)
    correctness_history: List[bool] = field(default_factory=list)
    visit_count: int = 0
    correct_streak: int = 0          # consecutive correct-with-conf≥6 answers
    wrong_streak: int = 0            # consecutive wrong answers while in Zone B
    last_answer_correct: bool = False
    last_confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.topic}::{self.question_type}::{self.difficulty_tier}"

    @property
    def confidence_avg(self) -> float:
        if not self.confidence_history:
            return 0.0
        return round(sum(self.confidence_history) / len(self.confidence_history), 2)

    def update(self, is_correct: bool, confidence: float) -> str:
        """
        Apply one answer to the node. Compute new zone via transition rules.
        Returns the new zone string.
        """
        # Rolling history (last 10)
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history = self.confidence_history[-10:]

        self.correctness_history.append(is_correct)
        if len(self.correctness_history) > 10:
            self.correctness_history = self.correctness_history[-10:]

        self.visit_count += 1
        self.last_answer_correct = is_correct
        self.last_confidence = confidence
        self.last_updated = time.time()

        # ── Zone transition logic ──────────────────────────────────────────
        if is_correct:
            self.wrong_streak = 0
            # Correct answer: increment high-confidence streak
            if confidence >= 6:
                self.correct_streak += 1
            else:
                self.correct_streak = 0

            if self.zone == ZONE_C:
                # Any correct answer lifts Zone C → Zone B
                self.zone = ZONE_B
                # Align with B→Green streak threshold (6+) so one good answer counts toward green
                self.correct_streak = 1 if confidence >= 6 else 0

            elif self.zone == ZONE_B:
                # Green: two confident corrects in a row, OR one very confident correct (8+)
                if self.correct_streak >= 2 or (
                    self.correct_streak >= 1 and confidence >= 8
                ):
                    self.zone = ZONE_GREEN

            elif self.zone == ZONE_GREEN:
                # Stay green; streak continues
                pass

        else:
            # Wrong answer: reset streak
            self.correct_streak = 0
            self.wrong_streak += 1

            if self.zone == ZONE_GREEN:
                # Buffered regression: a single wrong answer in green drops to Zone B.
                self.zone = ZONE_B
                self.wrong_streak = 1

            elif self.zone == ZONE_B:
                # Escalate to Zone C only after repeated wrong answers in Zone B.
                if self.wrong_streak >= 2:
                    self.zone = ZONE_C

            elif self.zone == ZONE_C:
                # Stay Zone C
                pass

        return self.zone


# Representative starter node definitions
_STARTER_NODES = [
    ("math",     "reasoning", "moderate"),
    ("math",     "reasoning", "hard"),
    ("math",     "factual",   "moderate"),
    ("math",     "reasoning", "expert"),
    ("code",     "code",      "moderate"),
    ("code",     "code",      "hard"),
    ("code",     "reasoning", "expert"),
    ("logic",    "reasoning", "moderate"),
    ("logic",    "reasoning", "hard"),
    ("logic",    "reasoning", "expert"),
    ("factual",  "factual",   "moderate"),
    ("factual",  "factual",   "hard"),
    ("factual",  "reasoning", "moderate"),
    ("factual",  "factual",   "expert"),
    ("planning", "reasoning", "moderate"),
    ("planning", "reasoning", "hard"),
    ("planning", "factual",   "hard"),
    ("planning", "reasoning", "expert"),
    ("code",     "factual",   "moderate"),
    ("logic",    "factual",   "hard"),
]


class CalibrationMap:
    # Tiers that should start as zone_c (overconfident failure zones)
    _ZONE_C_TIERS = {"expert", "extreme"}

    def __init__(self):
        self.nodes: Dict[str, CalibrationNode] = {}
        self._seed_nodes()

    def _seed_nodes(self):
        """
        Seed 20 representative starter nodes.
        expert/extreme tiers start in zone_c (high-confidence failure targets).
        moderate/hard tiers start in zone_b.
        """
        for topic, qtype, tier in _STARTER_NODES:
            node = CalibrationNode(
                topic=topic,
                question_type=qtype,
                difficulty_tier=tier,
                zone=ZONE_C if tier in self._ZONE_C_TIERS else ZONE_B,
            )
            self.nodes[node.key] = node

    def update_node(
        self,
        topic: str,
        question_type: str,
        difficulty_tier: str,
        is_correct: bool,
        confidence: float,
    ) -> str:
        """
        Get-or-create node; apply one answer; return new zone string.
        NOTE: zone is now computed inside node.update() — callers must not pass it.
        """
        difficulty_tier = str(difficulty_tier)
        node = CalibrationNode(
            topic=topic,
            question_type=question_type,
            difficulty_tier=difficulty_tier,
        )
        key = node.key
        if key not in self.nodes:
            self.nodes[key] = node
        return self.nodes[key].update(is_correct=is_correct, confidence=confidence)

    def get_zone_c_nodes(self) -> List[CalibrationNode]:
        """Zone C nodes sorted by avg confidence descending (most dangerous first)."""
        zone_c = [n for n in self.nodes.values() if n.zone == ZONE_C]
        return sorted(zone_c, key=lambda n: n.confidence_avg, reverse=True)

    def get_zone_b_nodes(self) -> List[CalibrationNode]:
        """All Zone B nodes."""
        return [n for n in self.nodes.values() if n.zone == ZONE_B]

    def get_zone_counts(self) -> dict:
        """Returns {zone_c, zone_b, green} counts."""
        return {
            "zone_c": sum(1 for n in self.nodes.values() if n.zone == ZONE_C),
            "zone_b": sum(1 for n in self.nodes.values() if n.zone == ZONE_B),
            "green":  sum(1 for n in self.nodes.values() if n.zone == ZONE_GREEN),
        }

    def to_dict(self) -> dict:
        """Returns serialisable dict for JSON API — full node shape for frontend."""
        return {
            "nodes": [
                {
                    "key":              n.key,
                    "topic":            n.topic,
                    "question_type":    n.question_type,
                    "difficulty_tier":  n.difficulty_tier,
                    "zone":             n.zone,
                    "visits":           n.visit_count,
                    "visit_count":      n.visit_count,          # backwards compat
                    "avg_confidence":   n.confidence_avg,
                    "confidence_avg":   n.confidence_avg,       # backwards compat
                    "correct_streak":   n.correct_streak,
                    "wrong_streak":     n.wrong_streak,
                    "last_answer_correct": n.last_answer_correct,
                    "last_confidence":  n.last_confidence,
                }
                for n in self.nodes.values()
            ]
        }

    def snapshot(self) -> dict:
        """to_dict() plus timestamp and zone counts — used by WebSocket stream."""
        base = self.to_dict()
        base["timestamp"] = time.time()
        counts = self.get_zone_counts()
        base["zone_c_count"] = counts["zone_c"]
        base["zone_b_count"] = counts["zone_b"]
        base["green_count"]  = counts["green"]
        return base

    def reset(self):
        """Reset all nodes back to seed state."""
        self.nodes = {}
        self._seed_nodes()
