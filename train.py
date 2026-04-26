# ============================================================
# EVOAI LAB — COMPLETE TRAINING IN ONE CELL
# Run this, wait 30-40 mins, done.
# ============================================================

# Step 1: Install
import subprocess
subprocess.run(["pip", "install", "-q", "--upgrade",
    "transformers", "peft", "accelerate", "datasets",
    "huggingface_hub", "trl"], check=True)

print("Packages installed")

# Step 2: Verify
import torch
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

assert torch.cuda.is_available(), "No GPU — go to Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))

# Step 3: Login (Colab: add HF_TOKEN in Secrets, or: os.environ["HF_TOKEN"] = "...")
import os
from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN (e.g. Colab User secrets or export before running).")
login(token=HF_TOKEN)

# Step 4: Load model
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (5-8 mins)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("Model loaded:", round(model.num_parameters()/1e9,1), "B params")

# Step 5: Apply LoRA
lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_cfg)
model.config.use_cache = False
model.print_trainable_parameters()

# Step 6: Dataset
def fmt(q, a):
    return {
        "prompt": f"<s>[INST] {q} [/INST]",
        "completion": f"<s>[INST] {q} [/INST] {a}</s>",
    }

dataset = Dataset.from_list([
    fmt("Is 0.1 + 0.2 == 0.3 in Python?",
        "No. 0.1 + 0.2 evaluates to 0.30000000000000004 due to floating point precision. Use math.isclose() instead."),
    fmt("What does list.sort() return?",
        "None. list.sort() sorts in place and returns None. Use sorted(my_list) to get a new sorted list."),
    fmt("Can you use a list as a dict key?",
        "No. Lists are mutable and not hashable so they cannot be dict keys. Use a tuple instead."),
    fmt("Does Python's GIL prevent all parallelism?",
        "No. The GIL only blocks CPU-level threads. I/O-bound tasks still benefit from threading. Use multiprocessing for CPU-bound parallelism."),
    fmt("Is a mutable default argument safe in Python?",
        "No. Default arguments are created once at definition time not per call. Use None as default and initialize inside the function."),
    fmt("Does range(10) create a list in Python 3?",
        "No. range() returns a lazy range object not a list. Call list(range(10)) to get an actual list."),
    fmt("What is the difference between == and is?",
        "== checks value equality. is checks object identity. Never use is to compare strings or integers, always use ==."),
    fmt("What does dict.update() return?",
        "None. dict.update() modifies the dict in place and returns None. Use {**d1, **d2} to create a merged dict."),
    fmt("Is it safe to delete from a list while iterating?",
        "No. Deleting during iteration shifts indices and silently skips elements. Use a list comprehension to filter instead."),
    fmt("Does Python pass integers by reference?",
        "No. Integers are immutable. Reassigning an integer inside a function does not affect the caller. Only mutable objects like lists can be mutated inside a function."),
])
print(f"Dataset: {len(dataset)} examples")

# Step 7: Reward function
def reward_fn(completions, prompts, **kwargs):
    rewards = []
    for completion in completions:
        score = 0.0
        text = completion.lower().strip()
        n = len(text.split())
        if text[:4] in ["no. ","yes.","no, ","yes,"]: score += 0.25
        if any(p in text for p in ["because","therefore","this means","since"]): score += 0.20
        if any(p in text for p in ["use ","instead","the correct","you can"]): score += 0.20
        if 20 <= n <= 100: score += 0.15
        if n < 8: score -= 0.50
        if any(p in text for p in ["it depends","this is complex","it varies"]): score -= 0.20
        rewards.append(float(score))
    return rewards

# Step 8: Train
config = GRPOConfig(
    output_dir="./evoai_output",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=4,
    max_new_tokens=150,
    temperature=0.8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    tokenizer=tokenizer,
)

print("\nStarting training — do not interrupt")
print("Expected time: 20-40 mins on T4")
print("="*45)
trainer.train()
print("\nTRAINING COMPLETE")

# Step 9: Test
def ask(q):
    inputs = tokenizer(f"<s>[INST] {q} [/INST]",
                      return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150,
                            temperature=0.7, do_sample=True,
                            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

print("\nTEST RESULTS")
print("="*45)
for q in ["Is 0.1 + 0.2 == 0.3 in Python?",
          "What does list.sort() return?",
          "Does Python pass integers by reference?"]:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")

# Step 10: Save metrics
import json
log = trainer.state.log_history
steps = [e["step"] for e in log if "loss" in e]
losses = [round(e["loss"],4) for e in log if "loss" in e]
rwds = [round(e.get("reward/mean",0),4) for e in log if "loss" in e]

print("\nTRAINING METRICS")
print(f"{'Step':>6}  {'Loss':>8}  {'Reward':>8}")
print("-"*30)
for s,l,r in zip(steps,losses,rwds):
    print(f"{s:>6}  {l:>8.4f}  {r:>+8.4f}")

if losses:
    print(f"\nLoss  : {losses[0]} → {losses[-1]}")
    print(f"Reward: {rwds[0]:+.4f} → {rwds[-1]:+.4f}")

with open("metrics.json","w") as f:
    json.dump({"steps":steps,"losses":losses,"rewards":rwds},f,indent=2)
print("\nSaved metrics.json")
print("DONE — screenshot this output for your demo")