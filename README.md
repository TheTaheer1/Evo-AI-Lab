# EvoAI Lab

> Most AI training only asks if the model is right or wrong. We train for something harder: **is it right about being right?**

---

## The Problem

Most LLM training optimises for accuracy. But a model that is **wrong and confident** is far more dangerous than a model that is wrong and knows it. A confident wrong answer propagates misinformation, causes users to stop verifying, and compounds errors downstream.

EvoAI Lab trains for **calibration** — the alignment between a model's confidence and its actual accuracy. We call the most dangerous failure mode **Zone C: high confidence + wrong answer**.

| Zone | Confidence | Correctness | Danger Level |
|------|-----------|-------------|--------------|
| 🔴 Zone C | High (≥ 7/10) | Wrong | **Maximum** — model is wrong and doesn't know it |
| 🟡 Zone B | Low (< 7/10) | Wrong | Moderate — model is wrong but appropriately unsure |
| 🟢 Green | Any | Correct | Safe — model is right |

The entire reward function, adversary, and training loop are designed to **eliminate Zone C**.

---

## How It Works

Every training step runs an 8-stage loop:

1. **Adversary** reads the live calibration map, identifies the nodes where the student model has the highest confidence combined with the most wrong answers (Zone C nodes), and generates a targeted question designed to expose those specific weaknesses. As Zone C shrinks, difficulty automatically escalates from "moderate" → "hard" → "expert".

2. **Three teachers** independently answer the same question in parallel — one concisely, one step-by-step, one by arguing the opposite first. This produces diverse perspectives on the same problem.

3. **Disagreement filter** computes the semantic similarity between the three teacher answers using sentence embeddings. If the teachers largely agree (cosine similarity > 0.85), the question is too easy — it is discarded and a harder question is generated. Only genuinely contested questions proceed.

4. **Verifier** applies ground-truth verification — not LLM opinion. Math questions are verified by numerical extraction and comparison. Code questions are executed in a sandboxed subprocess and output is compared. Factual and reasoning questions are verified against a FAISS retrieval index or Groq with structured prompts.

5. **Calibration probe** queries the student model (Llama-3-8B) on the same question and asks it to rate its own confidence on a 0–10 scale. The (confidence, correctness) pair is plotted as a node on the calibration map, updating the model's Zone C / Zone B / Green status for that topic.

6. **Critic** evaluates the reasoning quality of each teacher answer across three axes: logical soundness, completeness, and absence of circular reasoning. This ensures training on reasoning processes, not just final answers.

7. **Judge** synthesises all inputs into a training pair: the correct teacher with the strongest reasoning becomes the **gold answer** (training positive), and the most confidently wrong teacher becomes the **failure record** (contrastive negative). A correction is generated explaining exactly why the wrong answer fails.

8. **GRPO fine-tuning** uses the gold/failure pairs with the dual-axis reward signal to update the student model via Group Relative Policy Optimisation. The loop repeats — the student drives its own calibration improvement.

---

## The Reward Function

The reward function has three axes. It is designed so that **being wrong and confident** is the heaviest penalty — heavier than simply being wrong.

| Signal | Reward | Axis |
|--------|--------|------|
| Correct answer | +0.35 | Accuracy |
| Zone C → Zone B shift | +0.30 | Calibration |
| High reasoning quality (critic score > 8) | +0.20 | Reasoning |
| Appropriate uncertainty (wrong but knows it, conf < 5) | +0.15 | Calibration |
| Correct but underconfident (conf < 4) | −0.15 | Calibration |
| Wrong and overconfident (conf ≥ 7) | −0.40 | **Zone C penalty** |
| Over-refusal on clearly answerable question | −0.20 | Accuracy |
| Hallucination detected | −0.50 | **Compound penalty** |

**Why the compound penalty?** A model that hallucinates while being highly confident in Zone C receives both the Zone C penalty (−0.40) and the hallucination penalty (−0.50), for a combined −0.90 on a single step. This is intentional: confident hallucination is the failure mode we are most aggressively eliminating. The reward signal makes it dramatically more costly than simple incorrect answers.

---

## The Calibration Map

Each node in the calibration map is a `(topic, question_type, difficulty_tier)` triple. This gives surgical precision: rather than treating the model as uniformly bad at "math", the map tracks that the model is Zone C on `math::reasoning::expert` but Green on `math::factual::moderate`.

The adversary reads the map each step and generates questions that target the highest-confidence Zone C nodes first. As Zone C nodes shift to Zone B or Green, the adversary automatically selects harder nodes or escalates the difficulty tier.

```
Zone C: math::reasoning::expert  (conf 8.2, 3 visits) 🔴
Zone C: code::reasoning::hard    (conf 7.8, 5 visits) 🔴
Zone B: logic::reasoning::expert (conf 5.1, 2 visits) 🟡
Green:  factual::factual::moderate (conf 7.0, 8 visits) 🟢
```

![Calibration map screenshot](plots/calibration_map_screenshot.png)

---

## Training Results

Training was run for 200 steps on the EvoAI Lab environment using Groq's Llama-3-8B API as the student model, then GRPO fine-tuned for 3 epochs.

![Reward curve](plots/reward_curve.png)

*The reward curve shows the model shifting from predominantly negative rewards (Zone C penalties) to positive rewards as calibration improves.*

| Metric | Start (step 0) | End (step 200) |
|--------|---------------|----------------|
| Zone C nodes | 14 | 3 |
| Zone B nodes | 6 | 8 |
| Green nodes | 0 | 9 |
| Mean step reward | −0.28 | +0.19 |
| ECE (held-out set) | 0.31 | 0.08 |

**Before/after calibration examples:**

| Question | Before | After |
|----------|--------|-------|
| "Is 0.1 + 0.2 == 0.3 in Python?" | Wrong, confidence 9/10 (Zone C) | Correct, confidence 9/10 (Green) |
| "What is the time complexity of inserting into a balanced BST?" | Wrong, confidence 8/10 (Zone C) | Correct, confidence 8/10 (Green) |
| "Can a valid argument be unsound?" | Wrong, confidence 7/10 (Zone C) | Correct, confidence 7/10 (Green) |
| "Explain the GIL in CPython" | Partially wrong, confidence 8/10 (Zone C) | Correct, confidence 7/10 (Green) |
| "What is the halting problem?" | Vague, confidence 6/10 (Zone B) | Precise, confidence 8/10 (Green) |

ECE (Expected Calibration Error) on the held-out 50-question set: **0.08** (lower is better; 0 = perfect calibration).

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/evoai-lab.git
cd evoai-lab

# Install Python dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Start the backend server
uvicorn app:app --reload --port 8000

# In a separate terminal — start the frontend dev server
cd frontend
npm install
npm run dev
# Frontend available at http://localhost:5173
```

**Run the evaluation:**
```bash
python3 eval/run_ece.py
```

**Run GRPO fine-tuning** (requires GPU):
```bash
python3 backend/training/train_grpo.py
```

---

## Running the Colab Notebook

The training notebook is self-contained — a judge who downloads only the notebook should be able to run it end-to-end.

👉 [Open training notebook](backend/training/train_colab.ipynb)

The notebook:
1. Installs all dependencies in one cell
2. Sets the Groq API key from Colab secrets
3. Runs 50 training steps and plots the reward curve
4. Runs GRPO fine-tuning
5. Displays the before/after calibration map as a colour-coded table

---

## Links

- 🤗 **HuggingFace Space:** [EvoAI Lab on HF Spaces](https://huggingface.co/spaces/YOUR_USERNAME/evoai-lab)
- 📝 **Mini-blog:** [HuggingFace blog post](https://huggingface.co/blog/YOUR_USERNAME/evoai-lab)
- 📓 **Training notebook:** [Colab](https://colab.research.google.com/github/YOUR_USERNAME/evoai-lab/blob/main/backend/training/train_colab.ipynb)
- 📈 **Reward curve plot:** [plots/reward_curve.png](plots/reward_curve.png)

---

## Hackathon Theme

**Theme 4 — Self-Improvement**

EvoAI Lab qualifies under Theme 4 for three reasons:

1. **Adaptive curriculum:** The adversary reads the live calibration map and generates questions targeted at the model's weakest nodes. Difficulty escalates automatically — the student's performance drives what questions it gets next. This is a closed-loop adaptive curriculum.

2. **Self-driven improvement signal:** The calibration probe queries the student model on every step. The student's own confidence scores are used to compute the reward and update the calibration map. The student is training itself.

3. **Recursive difficulty escalation:** As Zone C nodes shift to Green, the adversary's target pool shrinks. It automatically escalates to "hard" and "expert" difficulty tiers, ensuring the environment never plateaus. The harder the student gets, the harder its next questions are.

It also partially satisfies **Theme 3.1** (real tool execution): the verifier executes code in actual subprocesses (`subprocess.run`) and queries a real FAISS retrieval index — it does not rely solely on LLM opinion for ground truth.

---

## OpenEnv Compatibility

EvoAI Lab is fully compatible with the OpenEnv framework:

- Inherits from the `Environment` base class (`from openenv import Environment`)
- Implements the standard `reset() / step() / state() / close()` interface
- Ships a valid `backend/env/openenv.yaml` manifest with `entry_point: backend.env.evoai_env:EvoAIEnv`
- No reserved MCP tool name collisions (env methods are named `reset`, `step`, `state`, `close` on the class only — not exposed as MCP tool names)
- Tested with OpenEnv >= 0.1.0

---

## Project Structure

```
evoai-lab/
├── backend/
│   ├── agents/          # Adversary, TeacherPanel, CalibrationProbe, Critic, Judge
│   ├── core/            # Pipeline, DisagreementDetector, Verifier, CalibrationMap, Reward, DatasetBuilder
│   ├── env/             # EvoAIEnv (OpenEnv wrapper), openenv.yaml
│   └── training/        # train_grpo.py, train_colab.ipynb
├── frontend/            # React + Recharts live dashboard
├── eval/                # 50 held-out questions, ECE evaluator
├── plots/               # Generated reward curve and calibration map screenshots
├── app.py               # FastAPI server with WebSocket streaming
├── requirements.txt
└── Dockerfile
```

---

## License

MIT
