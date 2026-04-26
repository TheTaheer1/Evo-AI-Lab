"""
disagreement.py
Compute semantic similarity between teacher answers and filter out
questions where teachers agree (avg cosine similarity > threshold).
"""
import numpy as np

from backend.core.text_encoder import get_sentence_transformer


class DisagreementDetector:
    def __init__(self, threshold: float = 0.70):
        self._model = None
        self.threshold = threshold
        self.total_seen = 0
        self.total_passed = 0

    def _encoder(self):
        if self._model is None:
            self._model = get_sentence_transformer()
        return self._model

    def filter(self, teacher_outputs: list) -> bool:
        """
        Returns True if teachers disagree enough to keep the question.
        Returns False if teachers largely agree (question is too easy / trivial).
        """
        answers = [t.get("answer", "") for t in teacher_outputs]
        # Guard: need at least 2 answers
        if len(answers) < 2:
            self.total_seen += 1
            self.total_passed += 1
            return True

        # Encode all answers (lazy model load on first filter)
        embeddings = self._encoder().encode(answers, convert_to_numpy=True, normalize_embeddings=True)

        # Compute pairwise cosine similarities for all pairs
        n = len(embeddings)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                similarities.append(sim)

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        self.total_seen += 1

        if avg_sim > self.threshold:
            # Teachers agree — skip this question
            return False

        self.total_passed += 1
        return True

    @property
    def pass_rate(self) -> float:
        return self.total_passed / max(self.total_seen, 1)
