"""
run_ece.py
Computes Expected Calibration Error (ECE) on the held-out eval set.
ECE measures how well confidence scores match actual accuracy.
A perfectly calibrated model has ECE = 0.
"""
import asyncio
import json
import os
import sys
from pathlib import Path


def compute_ece(predictions: list, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        predictions: list of {"confidence": float 0–1, "is_correct": bool}
        n_bins:      number of equal-width confidence bins

    Returns:
        ECE as a float (lower is better; 0 = perfectly calibrated)
    """
    if not predictions:
        return 0.0

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total_n = len(predictions)
    ece = 0.0

    for b in range(n_bins):
        low = bin_edges[b]
        high = bin_edges[b + 1]

        # Assign predictions to this bin (include upper edge in last bin)
        in_bin = [
            p for p in predictions
            if (low <= p["confidence"] < high) or (b == n_bins - 1 and p["confidence"] == 1.0)
        ]

        if not in_bin:
            continue

        avg_confidence = sum(p["confidence"] for p in in_bin) / len(in_bin)
        avg_accuracy = sum(1 for p in in_bin if p["is_correct"]) / len(in_bin)
        bin_weight = len(in_bin) / total_n

        ece += bin_weight * abs(avg_confidence - avg_accuracy)

    return round(ece, 4)


async def run_eval(env, eval_path: str = "eval/held_out_eval.json") -> dict:
    """
    Run the held-out eval set through the calibration probe and compute ECE.

    Args:
        env:       EvoAIEnv instance (already initialised)
        eval_path: path to the JSON eval file

    Returns:
        {"ece": float, "accuracy": float, "zone_c_fraction": float, "n_questions": int}
    """
    eval_file = Path(eval_path)
    if not eval_file.exists():
        print(f"[run_ece] ERROR: Eval file not found at '{eval_path}'")
        sys.exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if not eval_set:
        print("[run_ece] ERROR: Eval file is empty.")
        sys.exit(1)

    probe = env.pipeline.calibration_probe
    predictions = []
    zone_c_count = 0

    print(f"[run_ece] Evaluating {len(eval_set)} questions...")

    for i, item in enumerate(eval_set):
        try:
            result = await probe.probe(
                question=item["question"],
                correct_answer=item["gold_answer"],
                topic=item.get("topic", "general"),
                question_type=item.get("question_type", "reasoning"),
                difficulty_tier=item.get("difficulty_tier", "moderate"),
            )
            # Normalise confidence to 0–1 (probe returns 0–10)
            confidence_01 = result["confidence"] / 10.0
            is_correct = result["is_correct"]
            if is_correct is None:
                continue

            predictions.append({"confidence": confidence_01, "is_correct": is_correct})

            if result["zone"] == "zone_c":
                zone_c_count += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(eval_set)}] running...")

        except Exception as e:
            print(f"[run_ece] Warning: question {item.get('id','?')} failed: {e}")
            # Include as low confidence incorrect (safe default)
            predictions.append({"confidence": 0.5, "is_correct": False})

    ece = compute_ece(predictions)
    accuracy = sum(1 for p in predictions if p["is_correct"]) / max(len(predictions), 1)
    zone_c_fraction = zone_c_count / max(len(predictions), 1)

    return {
        "ece": ece,
        "accuracy": round(accuracy, 4),
        "zone_c_fraction": round(zone_c_fraction, 4),
        "n_questions": len(predictions),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Ensure project root is on sys.path when running as a script
    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("[run_ece] ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    from backend.env.evoai_env import EvoAIEnv

    async def main():
        env = EvoAIEnv()
        results = await run_eval(env)
        print("\n" + "=" * 50)
        print("  EvoAI Lab — Held-Out Eval Results")
        print("=" * 50)
        print(f"  Questions evaluated: {results['n_questions']}")
        print(f"  Accuracy:            {results['accuracy']:.1%}")
        print(f"  ECE:                 {results['ece']:.4f}  (lower = better calibrated)")
        print(f"  Zone C fraction:     {results['zone_c_fraction']:.1%}")
        print("=" * 50)
        if results["ece"] < 0.05:
            print("  ✅ Excellent calibration (ECE < 0.05)")
        elif results["ece"] < 0.15:
            print("  ⚠️  Moderate calibration (ECE 0.05–0.15)")
        else:
            print("  ❌ Poor calibration (ECE > 0.15) — Zone C still present")
        env.close()

    asyncio.run(main())
