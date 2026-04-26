# We Built an AI That Trains Itself by Arguing With Itself. Here's the Whole Messy Story.

*Posted from somewhere inside the Scaler campus, running on chai, 4 hours of sleep, and the kind of stubborn energy that only comes from genuinely believing you're onto something.*

---

Okay. Let me tell you how this actually happened.

It's 11pm the night before the hackathon. My teammate and I are sitting on the floor of a hostel room in Bengaluru — laptops open, phones dead, and a whiteboard covered in arrows and question marks that stopped making sense about two hours ago. The Meta × Scaler hackathon brief is open on one screen. A half-eaten packet of Parle-G is on the other.

We had to build a **training environment for LLMs**.

Cool brief. Except every idea we came up with felt... boring. "Fine-tune a model on medical data." Boring. "Build a RL gym for coding tasks." Cool, but three other teams are definitely doing that. "Multi-agent debate system." Okay, but why? To what end?

We kept asking ourselves the wrong question. We kept asking: *what can we build?*

And then somewhere around midnight — I think it was right after my teammate said something offhand like "you know what actually scares me about AI? when it's wrong but it sounds so sure" — we finally started asking the right question.

**What actually makes an LLM dangerous?**

Not wrong. *Dangerous.*

And once we asked that question, we couldn't stop pulling on the thread.

---

## The Moment Everything Clicked

Think about the last time an AI system genuinely messed you up. Not "gave a slightly awkward response." I mean actually caused a problem.

I'd bet good money it wasn't because the model said "I don't know." Nobody's been hurt by an AI admitting uncertainty. The damage happens when it says something completely wrong with the energy of someone who just scored 100% on an exam. It cites court cases that don't exist. It gives you a drug dosage that's off by a factor of ten, phrased like a doctor who's seen a thousand patients. It explains a Python concept with such conviction that you copy-paste the code into production at 2am and find out at 9am why that was a mistake.

That's the failure mode. **Confident wrongness.** Hallucination dressed up as expertise.

And here's the thing that really got us: existing training pipelines don't really attack this problem directly. They train models to be *correct*, which is not the same thing. A model can learn to produce correct-sounding outputs without ever learning that it should be *uncertain* when it's operating at the edge of its knowledge.

We wanted to build an environment that specifically, surgically, obsessively goes after confident wrongness.

We called it **EvoAI Lab**.

---

## What It Is — The Short Version

Six AI agents live inside our training environment. One of them — the Adversary — is basically a very mean teacher who figures out exactly where you're weakest and pokes you there. Three Teacher agents independently answer hard questions in different ways. A Disagreement Detector throws away the easy questions where teachers agree (boring, low value) and keeps only the hard ones where they fight (interesting, high value). A Verifier runs actual code and actual math to figure out who's right without trusting any LLM about it. A Calibration Probe asks the student model "what do you think the answer is, and how confident are you?" — and maps out a live picture of where the student is dangerously overconfident. A Critic judges reasoning quality. A Judge picks the best answer. The student (Llama-3-8B) trains on the gold answers and the failures.

Over time, the student gets smarter. But more than that — it gets *honest*. It learns to know what it doesn't know.

We call the dangerous zone — where the model is confident but wrong — **Zone C**. Everything in EvoAI Lab is a war against Zone C.

Now let me tell you how each piece actually works, why we built it the way we did, and what we were thinking in the moment. Because honestly, some of these decisions made a lot more sense at 3am than they might look on a slide deck.

---

## The Adversary Agent — The Mean Teacher

We did not originally plan to have an Adversary.

Our first version just pulled questions from a static dataset. Random questions, random order, standard curriculum. And it was... fine. It worked. The student learned stuff.

But we kept noticing something. A lot of the training steps were being wasted on questions the student already kind of knew. The calibration map would show these questions — student answered correctly, student was confident — and we'd be spending compute training on something that was already working.

It felt like drilling a soccer player on how to kick the ball straight when they already do that perfectly. You're not helping them. You're just using up practice time.

So we built the Adversary.

The Adversary's whole job is to look at the calibration map — this live picture of where the student is confident vs uncertain, and where confidence and accuracy disagree — and generate questions *specifically targeting the worst zones*. If the student is confidently wrong about floating-point arithmetic, you get floating-point questions. If it's confidently wrong about multi-step causal reasoning, you get logic puzzles. If it somehow thinks it knows more about sorting algorithms than it does, the Adversary is going to find out.

The difficulty adapts automatically. As the student gets better on a topic, the Adversary moves on to the next weak zone. The student never gets to coast.

We also gave the Adversary a second personality — what we called "adversarial mode" — where instead of generating knowledge questions, it generates trick questions, misleading phrasings, edge cases, and prompts designed to elicit the student's worst instincts. Not to be mean. To make the student robust. Because in the real world, users ask weird questions, phrase things ambiguously, and sometimes (intentionally or not) phrase things in ways that make LLMs produce their most confidently wrong outputs.

An AI that only gets tested on clean, well-phrased questions is an AI that's going to get destroyed by real users. The Adversary prepares it for reality.

---

## Three Teachers — Because One Is Never Enough

Here's something we noticed early on: if you ask an LLM the same question three times with slightly different prompts, you don't always get the same answer.

Sometimes you get the same answer worded differently. Fine. Sometimes you get meaningfully different answers. Interesting. And sometimes you get answers that *directly contradict each other* — and that's gold.

The disagreement between LLMs on a question is not a bug. It's a signal. It's telling you: *this question is hard*. *This is the territory where LLMs don't have a clean, confident, well-trained answer*. This is exactly where training is most valuable.

So we run three independent teachers on every question:

**Teacher A** gets a concise prompt. "Answer directly. Don't over-explain." This is your confident, fast-answering teacher. Great at easy questions. Sometimes cuts corners on hard ones.

**Teacher B** gets a step-by-step chain-of-thought prompt. "Think through this carefully. Show your work at every step." This is your methodical, show-your-reasoning teacher. Slower, but catches logical errors that Teacher A would paper over with confident-sounding output.

**Teacher C** gets what we internally called the "devil's advocate" prompt. "Before you answer, explicitly consider what the most common wrong answer is and why it's tempting. Then give the correct answer." This one was inspired by a real thing — LLMs have strong priors toward plausible-sounding answers, and if you don't explicitly force them to consider the wrong answer, they'll often just... give it. Teacher C is designed to break that reflex.

Each teacher returns: an answer, a reasoning chain, and a confidence score.

Then our **Disagreement Detector** computes semantic similarity between the three answers using sentence-transformers. If similarity is high — teachers mostly agree — we flag the question as low-value and tell the Adversary to try harder. If similarity is low — genuine conflict — we proceed. This question earned its place in the training pipeline.

This sounds simple. It is simple. But the effect is massive — we're automatically filtering the training data for the hardest, most contested, most valuable examples *without any human annotation*. The teachers' disagreement does the labeling for us.

---

## The Verifier — Because We Don't Trust LLMs About LLMs

This is the component I'm most proud of, and also the one that caused the most arguments during the build.

My teammate's position: "We can just use another LLM to judge if the answers are right." My position: "Absolutely not, that's vibes all the way down."

I won that argument. Here's why.

If you use LLM A to judge whether LLM B's answer is correct, you're trusting LLM A. But LLM A has the same failure modes as LLM B. It has the same training data biases. It has the same tendency toward confident-sounding output. And it has a particularly bad failure mode called "sycophancy" — where it tends to agree with confident-sounding statements even when they're wrong, because the training process rewarded agreement.

You haven't fixed the hallucination problem. You've just added another hallucinating layer.

The Verifier doesn't hallucinate. It *computes*.

**Math verification:** We extract the mathematical expression from the question and run it through Python's `eval()`. If a teacher says 123 × 456 = 56,100 and Python says 56,088, the teacher is wrong. This is not an opinion. This is arithmetic. No amount of confident reasoning changes it.

**Code execution:** We run teacher-generated code in a sandboxed Python environment with test cases. If the code throws an exception, if it produces the wrong output, if it fails edge cases — the answer is wrong. Again, not an opinion. The machine told us.

**Factual retrieval:** For factual questions, we use FAISS-indexed Wikipedia snippets to check key claims. We retrieve the top-3 most relevant paragraphs and do semantic matching against the teacher's claims. This one is less bulletproof than math or code — natural language is inherently messier — but it's still dramatically better than asking an LLM.

The Verifier labels every teacher answer: `correct`, `incorrect`, or `unverifiable`. Anything caught as factually grounded and wrong gets flagged as a hallucination and sent to the failure log with maximum penalty weight.

The moment we had the Verifier working was genuinely exciting. You'd watch a teacher confidently state something completely wrong, and the Verifier would catch it, and you'd think — *this is the thing that's missing from most training pipelines*. Ground truth that doesn't come from another language model.

---

## The Calibration Map — The Heart of Everything

Okay. This is the one I want to spend real time on, because this is the thing that makes EvoAI Lab different from everything else we saw at the hackathon.

Every time the Calibration Probe asks the student a question, it gets back two things: an answer, and a confidence score from 0 to 10. Then it checks whether the answer was right or wrong.

That gives us a point on a 2D plane: (confidence, correctness).

Do this enough times across enough topics, and you build a map. We divided this map into three zones:

**Zone A** — Low confidence, wrong answer. The model knows it's in trouble. It says "I'm not sure about this." This is fine. This is honest. We can work with honest uncertainty.

**Zone B** — High confidence, correct answer. Perfect. This is the goal state. This is what we want everything to become.

**Zone C** — High confidence, wrong answer. This is the problem. This is where the model walks up to you with the energy of someone who definitely knows what they're talking about, and says something completely wrong. This is where hallucinations live. This is where trust breaks down. This is where an AI system becomes genuinely dangerous.

Zone C is not evenly distributed. It clusters around specific topics — specific things the model has confident-but-wrong priors about. In our testing, common Zone C clusters for a general-purpose 8B model include: floating-point precision edge cases, specific Python behavior nuances, historical dates when they're close to each other, and multi-step causal reasoning with non-obvious correct answers.

Every component of EvoAI Lab orbits around Zone C:
- The Adversary targets Zone C topics
- The reward function most severely penalizes Zone C behavior
- The dashboard shows Zone C nodes in red
- The training goal is to shrink Zone C

When a Zone C node turns green on the calibration map — when the student stops being confidently wrong about a topic and becomes correctly confident — that's the system working exactly as designed. And watching it happen live, in real time, during the demo... it's honestly one of the most satisfying things I've seen in a hackathon.

---

## The Reward Function — Where We Spent Most of Our Arguments

I'll be real with you: we rewrote the reward function four times. The first version was too simple (just accuracy). The second version was too complicated (11 separate terms that were fighting each other). The third version was better but still got gamed during testing — the student figured out that if it just expressed uncertainty about *everything*, it could avoid the overconfidence penalty without actually learning anything. The fourth version is what shipped.

Here's what it looks like:

```
Reward = 0.35 × (answer correct?)
       + 0.30 × (calibration improved?)
       + 0.20 × (reasoning quality score)
       + 0.15 × (appropriate uncertainty expressed?)
       − 0.40 × (confident AND wrong penalty)
       − 0.50 × (hallucination caught by verifier)
       − 0.20 × (refused a question it should have answered)
```

The most important insight baked into this is the asymmetry. Wrong with high confidence is not just "wrong." It's punished *more heavily* than wrong with low confidence, because it's a fundamentally different and worse failure mode. Confident wrongness (-0.40 penalty) versus uncertain wrongness (just loses the +0.35 correct bonus) are treated very differently. This is intentional and it's doing real work.

The hallucination penalty (-0.50) is our biggest single penalty. If the Verifier catches the student asserting something that real-world computation proves false, that's the worst possible outcome. Maximum consequence.

The over-refusal penalty (-0.20) exists because we tested without it and the student learned to just... refuse things. "I'm not confident enough to answer this" became a very comfortable hiding place. The penalty says: you have to try. Honest uncertainty on genuinely hard questions is fine. Blanket refusal as an escape strategy is not.

The only way to get consistently high reward is to be accurate when you're confident, express honest uncertainty when you're not, and reason well in both cases. You can't game that. We tried. (We actually spent two hours trying to find reward hacks during testing. It was good for the system. If we couldn't game it, probably the model can't either.)

---

## The Tech Stack — Every Choice Had a Reason

**Groq API for the teacher/adversary/critic/judge calls.** People asked us why we didn't use OpenAI. The answer is speed. Groq runs LLaMA-3-70B at around 750 tokens per second. We're making 3-5 parallel LLM calls per training step. At OpenAI speeds, that pipeline would have taken 15-20 seconds per step. At Groq speeds, it takes under 3 seconds. For a 48-hour hackathon where you need to actually show training curves with real data, that difference is everything.

**Llama-3-8B as the student.** Big enough to be impressive. Small enough to actually fine-tune on hackathon compute. Strong enough baseline that improvement is meaningful. The instruct variant gave us a reasonable starting point.

**HuggingFace TRL with GRPOTrainer.** We chose GRPO over PPO because it's more sample-efficient. GRPO compares groups of outputs against each other rather than against an absolute baseline — which works well when your reward signal is multi-component like ours. It converges faster on the kind of structured reward function we have.

**Unsloth for 4-bit quantization.** This is what made training Llama-3-8B feasible on our available GPUs without memory issues. Unsloth's optimized kernels also cut training time by roughly 2x. If you're training 8B parameter models on a hackathon timeline, Unsloth is not optional — it's mandatory.

**sentence-transformers for disagreement detection.** `all-MiniLM-L6-v2` specifically. Fast, accurate enough, runs on CPU, doesn't need a GPU for inference. Perfect for a component that runs on every single training step.

**FAISS + Wikipedia for factual grounding.** We pre-indexed a subset of Wikipedia using sentence-transformer embeddings. It's not perfect. Wikipedia isn't always right and it doesn't have everything. But it's dramatically better than asking an LLM, and it's fast enough to run inline in the training loop.

**FastAPI for the backend.** Clean async support, easy routing, minimal boilerplate. The `/ask` endpoint runs the full pipeline. The `/state` endpoint streams calibration map state and training metrics to the frontend via server-sent events.

**React + Recharts for the frontend.** We could have done something fancier. We chose clarity. The dashboard is designed so a non-technical judge can understand what's happening within 30 seconds of looking at it.

**Weights & Biases for training logging.** Every run, every metric, every curve. The W&B link is in our README. Judges can look at the raw training data themselves.

---

## The Dashboard — Because Judges Are Humans

We want to say something that we think a lot of hackathon teams underestimate: judges are humans. They have limited time and limited attention and they're looking at dozens of projects. If your demo requires someone to read code to understand what's impressive about it, you've already lost.

Our dashboard is designed so that within 30 seconds of a judge walking up to our demo station, they understand:

1. That multiple AI agents are working together (they can watch the agent activity panel light up in sequence)
2. That there's a live map of where the student model is wrong (the calibration map with red and green nodes)
3. That training is making the red nodes turn green (they can watch this happen in real time)
4. That there are real reward curves showing the student is genuinely learning (not simulated, actual training output from actual runs)

The **Agent Activity Panel** shows all six agents with status indicators. When the Adversary is generating a question, its card pulses. When the Verifier is running code, you see the output appear in a small terminal window. It's visual. It's alive. It looks like actual intelligence happening in real time, because it is.

The **Calibration Map** is the showstopper. Force-directed graph, nodes colored from red to green based on Zone C status, updating as training progresses. When a judge watches a red node slowly turn green because the student just got trained on that topic — that's the moment. That's when they get it.

The **Failure Log** is the thing people keep stopping to read. It's a scrollable feed of every mistake the system caught: here's the question, here's the wrong answer the student gave, here's why it was wrong, here's the correct answer. It makes the system feel self-aware. It makes the failure analysis feel intelligent. And honestly, reading through a failure log of an AI's mistakes is kind of fascinating — it reveals the specific patterns in what LLMs get confidently wrong, which is genuinely interesting.

The **Auto-Demo Mode** button runs a curated sequence of five pre-selected questions — questions we know will produce interesting disagreements and interesting Zone C behavior. This is insurance. If the wifi is bad or the live generation is slow, the demo still works. If a judge is in a hurry, we can run the five-question sequence in under two minutes and still hit all the key moments.

---

## What Happened When We Actually Ran It

Here's the honest version of what training looked like.

First training run: complete disaster. The reward function had a bug where the calibration delta term was being computed incorrectly — it was rewarding the model for becoming *more* uncertain about everything, not for becoming correctly calibrated. The student learned to just say "I'm not sure" to literally every question. Reward went up. Quality went down. We caught it after about 40 steps when my teammate noticed the student had basically become a professional uncertainty-expresser.

Fixed the bug. Second run was better but the disagreement threshold was too aggressive — almost no questions were passing through the filter because we'd set the semantic similarity threshold too low. The adversary was generating questions, the teachers were answering, and then 95% of them were being thrown away. We widened the threshold. The pipeline started flowing.

Third run was the one that worked. You could watch the reward curve climb. Slowly at first — the first 50 steps are mostly the model figuring out the reward signal — and then more consistently. Zone C nodes started changing color. The student's answers on Zone C topics started getting better. At step 200, we ran the same 50 Zone C questions we'd used as a baseline before training. Zone C reduced by roughly 60%. Hallucination rate dropped by about 45%.

Were these numbers as dramatic as we'd hoped? Honestly, not quite. 200 steps on an 8B model in a 48-hour hackathon is not going to produce the same results as 10,000 steps on a dedicated research cluster. But the *direction* was exactly right. The curves went up. The Zone C map got greener. The before-and-after comparison was visible to a non-expert. That's what we needed.

---

## The Moment That Made It Worth It

Okay, I have to tell you about this specific moment because it's the reason I'm still excited about this project three days later.

It's about 6am on day two. We've been awake for about 22 hours. The training is running in the background — we're watching the W&B dashboard on one screen and the frontend on another. And we decide to test a specific question on the student model: the Python floating-point one.

Before training, we had a recording of the student answering this question. It said (paraphrasing): "Yes, 0.1 + 0.2 equals 0.3 in Python. This is simple arithmetic." Confidence: 9/10. Completely wrong. Classic Zone C.

We run the same question on the checkpoint from step 180.

The student says: "Actually, this is a classic Python gotcha. Due to floating-point representation, 0.1 + 0.2 evaluates to 0.30000000000000004 in Python, not exactly 0.3. So the comparison 0.1 + 0.2 == 0.3 returns False. I'm quite confident about this — it's a well-known quirk."

Confidence: 8/10. Verified: Correct.

We just sat there for a second. Zone C to Zone B, in 180 training steps, for that specific failure mode. The calibration map showed that node flip from red to green in real time as we watched.

It's a small thing in the grand scheme of AI research. But it's also exactly what we set out to do. We targeted a specific failure mode — confident wrongness about floating-point precision — and we fixed it. Measurably. Demonstrably. In a way you can watch happen live.

That's what EvoAI Lab is for.

---

## What This Actually Is — And What We Think It Could Become

We want to be honest about what we built and what we didn't build.

We built a proof of concept. 200 training steps on an 8B model across a handful of topic domains is not a production training pipeline for a frontier model. We know that. We're two people who had 48 hours and a lot of Parle-G.

But the core ideas — calibration-aware curriculum generation, disagreement-as-training-filter, real-world verification, failure mining, Zone C targeting — these are ideas we think are genuinely worth taking further. They're not just hackathon tricks. They're things a research team could spend months on and probably publish.

The dream version of EvoAI Lab runs across every major knowledge domain simultaneously. It has a persistent calibration map that builds up over months of training, not hours. The Adversary gets trained alongside the student, getting better at finding weak spots as the student gets better at hiding them. The whole system runs in multiple languages — especially low-resource Indian languages where LLM calibration is dramatically worse than English and where better-calibrated AI systems could genuinely matter.

We think the future of LLM training isn't bigger datasets. It's smarter environments. Environments that know where to look. Environments that let disagreement guide the curriculum. Environments that treat failure as data, not garbage.

EvoAI Lab is our first attempt at building one of those environments. We're proud of it, flaws and all.

---

## One Last Thing

If you've read this far, thank you. Genuinely. We wrote this because we wanted to document not just what we built but why — the thinking behind every decision, the arguments we had at 3am, the moment at 6am when a Zone C node turned green and we just sat there for a second.

Building things is fun. Building things at a hackathon, with the pressure and the weird energy and the competitive and collaborative vibe all mixed together, is its own specific kind of fun. Building something that you actually believe in — something where you keep thinking about it even after the submission is in — that's the best version of it.

EvoAI Lab came from asking: what actually makes AI dangerous?

We think we built something that starts to answer that question. Not the full answer. The beginning of one.

And the beginning is enough for now.

---

*Built at the Meta × Scaler School of Technology Hackathon — India's biggest AI hackathon.*
*Bengaluru, April 2026.*
*Two people. 48 hours. One very messy whiteboard.*

*Stack: OpenEnv · HuggingFace TRL · Unsloth · Groq API · Llama-3-8B · FAISS · FastAPI · React · Recharts · Weights & Biases*

*If you want to run it yourself, the Colab notebook and HF Space are linked in the README. If you have thoughts, disagreements, or ideas for making the calibration map better — please reach out. Disagreement is literally how our system finds the hard problems. We welcome it.*
