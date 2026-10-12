# %% [markdown]
# # Lab 3.3 · LoRA-tune a small instruct model for ticket triage
#
# Day 3 · Session 3 (LLM adaptation with LoRA) · **25 min** · GPU runtime recommended (Colab T4)
#
# **What you'll build**
# - A before/after evaluation of Qwen2.5-0.5B-Instruct on a fixed task: ticket → `{"intent", "category", "priority"}` JSON.
# - Chat-formatted SFT data with **loss masking** done by hand (prompt tokens get label `-100`).
# - A LoRA adapter (~35 MB, vs. ~2 GB for the fp32 base) saved to `artifacts/lora-qwen-ticket/`.
#
# Training ~1,500 examples takes about 8–10 min on a T4. If time is short, set `ADAPTER_FALLBACK` (below) to the instructor-trained adapter and skip training.
#
# **Tasks**
# - [ ] 1 · Before: zero-shot JSON validity + intent accuracy on 100 tickets (4')
# - [ ] 2 · Build SFT records; apply template; mask prompt tokens (5')
# - [ ] 3 · Attach LoRA; check trainable %; start training (4')
# - [ ] 4 · While it trains: inspect a batch, explain the labels (6')
# - [ ] 5 · After: same 100 tickets; compare validity, accuracy, latency (5')
# - [ ] ✱ Merge the adapter; compare r = 4 vs. r = 32

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/applied-nlp.git"
    if not Path("/content/applied-nlp").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/applied-nlp"], check=True)
    os.chdir("/content/applied-nlp/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("nlp_utils.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
if not Path("data/tickets_train.csv").exists():
    subprocess.run([sys.executable, "prepare_data.py"], check=True)
sys.path.insert(0, os.getcwd())

import nlp_utils as U
U.set_seed(42)
print("device:", U.device(), "| quick mode:", U.QUICK)

# %%
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

pd.set_option("display.max_colwidth", 80)

BASE = U.MODELS["lora_base"]          # Qwen/Qwen2.5-0.5B-Instruct
ADAPTER_DIR = U.ARTIFACTS / "lora-qwen-ticket"
# Set to a local folder or Hub id of an instructor-trained adapter to skip training, e.g. "artifacts/lora-qwen-ticket"
ADAPTER_FALLBACK = None
USE_FP16 = torch.cuda.is_available()  # mixed precision only on a GPU; never on CPU

DEV = U.device()
tok = AutoTokenizer.from_pretrained(BASE)
# fp32 weights + fp16 autocast during training is the most robust combination on a T4; 0.5B fits easily.
base_model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(DEV)
LABELS = U.load_labels()
print(f"{BASE}: {sum(p.numel() for p in base_model.parameters()) / 1e6:.0f}M params on {DEV}")

# %% [markdown]
# ## 1 · Before: zero-shot JSON validity + accuracy (4')
#
# The task: read a ticket, answer with **compact JSON** with three fields. The system prompt is short and fixed; we will use **exactly the same prompt** for training and inference. We measure:
# - **validity**: the output parses as JSON with the three keys (`json_valid`), and, stricter, has *only* those keys with values from our label sets (`schema_ok`),
# - **accuracy** per field (intent, category, priority), counting invalid outputs as wrong,
# - **latency** per ticket (greedy decoding).

# %%
SYSTEM = (
    "Classify the customer support ticket. Reply with JSON only, no other text: "
    '{"intent": "...", "category": "...", "priority": "..."}\n'
    f"intent: one of {', '.join(LABELS['intents'])}\n"
    f"category: one of {', '.join(LABELS['categories'])}\n"
    "priority: one of low, medium, high, urgent"
)
FIELDS = ["intent", "category", "priority"]


def chat_for(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def parse_json(raw):
    """First {...} block that parses and has the three keys, else None."""
    m = re.search(r"\{.*?\}", raw, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and all(k in obj for k in FIELDS) else None


@torch.no_grad()
def generate(model, text, max_new_tokens=40):
    enc = tok.apply_chat_template(chat_for(text), add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=tok.eos_token_id,
                         stop_strings=["}"], tokenizer=tok)   # stop after the first JSON object
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate(model, df):
    model.eval()
    rows = []
    for r in df.itertuples():
        t0 = time.perf_counter()
        raw = generate(model, r.text)
        ms = (time.perf_counter() - t0) * 1000
        obj = parse_json(raw) or {}
        schema_ok = (set(obj) == set(FIELDS) and obj["intent"] in LABELS["intents"]
                     and obj["category"] in LABELS["categories"] and obj["priority"] in LABELS["priorities"])
        rows.append({"ticket_id": r.ticket_id, "lang": r.lang, "text": r.text, "raw": raw, "valid": bool(obj),
                     "schema_ok": schema_ok,
                     "ms": ms, **{f: getattr(r, f) for f in FIELDS},
                     **{f"pred_{f}": str(obj.get(f, "INVALID")) for f in FIELDS}})
    return pd.DataFrame(rows)


def summarize(res):
    out = {"n": len(res), "json_valid": res.valid.mean(), "schema_ok": res.schema_ok.mean()}
    for f in FIELDS:
        out[f"{f}_acc"] = (res[f] == res[f"pred_{f}"]).mean()
    out["mean_ms"] = res.ms.mean()
    return out


test_df = U.load_tickets("test")
ar_df = U.load_tickets("ar_test")
eval_df = pd.concat([test_df.head(U.pick(100, 6)),
                     ar_df.sample(U.pick(20, 3), random_state=0)], ignore_index=True)
print("eval tickets:", eval_df.lang.value_counts().to_dict())

before = evaluate(base_model, eval_df)
pd.DataFrame({lang: summarize(g) for lang, g in before.groupby("lang")}).round(3)

# %%
before[["text", "raw", "valid"]].head(5)

# %% [markdown]
# 🔍 **Look at:** small instruct models often add prose around the JSON, invent intent names ("order_cancellation"), or mix up category and intent. A prompt alone can't fix that reliably. Teaching the format is exactly what LoRA is good at.

# %% [markdown]
# ## 2 · Build SFT records; apply template; mask prompt tokens (5')
#
# Each training record is the conversation we want to see: system prompt, the ticket, and the gold JSON as the assistant turn. We sample evenly across intents from the **training** split (never from test).

# %%
N_SFT = U.pick(1500, 48)
train_df = U.load_tickets("train")
per_intent = int(np.ceil(N_SFT / train_df.intent.nunique()))
sft_df = (train_df.sample(frac=1, random_state=42).groupby("intent").head(per_intent)
          .sample(frac=1, random_state=42).head(N_SFT).reset_index(drop=True))


def target_for(row):
    return json.dumps({"intent": row.intent, "category": row.category, "priority": row.priority}, ensure_ascii=False)


records = [{"messages": chat_for(r.text) + [{"role": "assistant", "content": target_for(r)}]}
           for r in sft_df.itertuples()]
print(len(records), "records")
print(json.dumps(records[0], ensure_ascii=False, indent=1)[-400:])

# %% [markdown]
# Now the key step. We want the loss **only on the assistant's answer**: the model should learn to produce the JSON, not to reproduce the system prompt and the ticket.
#
# ✏️ **Exercise:** write `build_example(msgs)` that returns `input_ids`, `attention_mask` and `labels`:
# 1. `prompt_ids` = chat template of `msgs[:-1]` with `add_generation_prompt=True` (ends with `<|im_start|>assistant\n`).
# 2. `full_ids` = chat template of the whole conversation.
# 3. Check that `prompt_ids` is a **prefix** of `full_ids` (if not, masking by length would be wrong).
# 4. Qwen's template ends the assistant turn with `<|im_end|>` **plus a newline**. Drop that trailing newline so the sequence ends with `<|im_end|>` (the token generation stops on).
# 5. `labels` = `-100` for every prompt token, then the answer token ids.
#
# In transformers v5, `apply_chat_template(..., tokenize=True)` (the default) returns a dict-like `BatchEncoding`, so take `["input_ids"]`.

# %%
def build_example(msgs, max_len=512):
    # BEGIN SOLUTION
    prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)["input_ids"]
    full_ids = tok.apply_chat_template(msgs)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt is not a prefix of the full conversation"
    if tok.decode(full_ids[-1:]) == "\n":       # template adds "\n" after <|im_end|>
        full_ids = full_ids[:-1]
    full_ids = full_ids[:max_len]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels[:len(full_ids)]}
    # END SOLUTION

# %% [markdown]
# Run the check below. If your function isn't finished yet, it falls back to a reference version so the rest of the lab still works (read it after you've tried).

# %%
try:
    ex = build_example(records[0]["messages"])
except NotImplementedError:
    print("build_example not implemented: using the reference version")

    def build_example(msgs, max_len=512):
        prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)["input_ids"]
        full_ids = tok.apply_chat_template(msgs)["input_ids"]
        assert full_ids[:len(prompt_ids)] == prompt_ids
        if tok.decode(full_ids[-1:]) == "\n":
            full_ids = full_ids[:-1]
        full_ids = full_ids[:max_len]
        labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[:len(full_ids)]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    ex = build_example(records[0]["messages"])

n_masked = sum(l == -100 for l in ex["labels"])
assert len(ex["input_ids"]) == len(ex["labels"]) and 0 < n_masked < len(ex["labels"])
assert ex["input_ids"][-1] == tok.convert_tokens_to_ids("<|im_end|>")
print(f"{len(ex['input_ids'])} tokens: {n_masked} masked (prompt), {len(ex['labels']) - n_masked} trained (answer)")
print("trained on:", repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])))

# %%
train_ds = Dataset.from_list([build_example(r["messages"]) for r in records])
lens = [len(x) for x in train_ds["input_ids"]]
print(train_ds, "| tokens per example: min", min(lens), "max", max(lens))


def collate(batch):
    """Right-pad to the longest example; padded positions get label -100 and attention 0."""
    width = max(len(x["input_ids"]) for x in batch)
    pad = lambda seq, value: seq + [value] * (width - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"], tok.pad_token_id) for x in batch]),
        "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch]),
        "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
    }

# %% [markdown]
# ## 3 · Attach LoRA; check trainable %; start training (4')
#
# LoRA freezes the base weights and learns a low-rank update `B·A` for each targeted linear layer. We target all attention and MLP projections (the slide's defaults). With `r=16` that is ~8.8M trainable parameters, under 2% of the model.

# %%
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
if ADAPTER_FALLBACK:
    model = PeftModel.from_pretrained(base_model, ADAPTER_FALLBACK)
    print("loaded adapter:", ADAPTER_FALLBACK)
else:
    model = get_peft_model(base_model, lora)
model.print_trainable_parameters()

# %% [markdown]
# Training settings: LoRA tolerates a much higher learning rate than full fine-tuning (2e-4). On a T4: batch 8 with gradient accumulation 2 (effective batch 16), one epoch. Quick mode (CPU) runs only a handful of steps with batch 2, enough to see the loss move, not to learn the task.

# %%
args = TrainingArguments(
    output_dir=str(U.CACHE / "runs" / "lora"), learning_rate=2e-4, num_train_epochs=1,
    per_device_train_batch_size=U.pick(8, 2), gradient_accumulation_steps=U.pick(2, 1),
    warmup_steps=0.05, lr_scheduler_type="cosine", logging_steps=U.pick(10, 2),
    max_steps=U.pick(-1, 6), fp16=USE_FP16, save_strategy="no", report_to="none",
    remove_unused_columns=False, seed=42)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)

if ADAPTER_FALLBACK is None:
    t0 = time.perf_counter()
    trainer.train()
    print(f"trained in {time.perf_counter() - t0:.0f}s")
    print(pd.DataFrame(trainer.state.log_history).dropna(subset=["loss"])[["step", "loss"]].round(3).to_string(index=False))
else:
    print("skipped training: using", ADAPTER_FALLBACK)

# %% [markdown]
# ## 4 · While it trains: inspect a batch, explain the labels (6')
#
# (In Jupyter this cell runs after training finishes; read it while the progress bar moves.)
#
# ✏️ **Exercise:** build a batch of 2 examples with `collate`. For each row print:
# 1. the number of real tokens, padded tokens and trained tokens (labels ≠ -100),
# 2. the decoded text of the tokens whose label is **not** -100.
#
# Explain to your neighbour: why is the system prompt not in the trained text? What would the model learn if it were?

# %%
# BEGIN SOLUTION
batch = collate([train_ds[i] for i in range(2)])
for row in range(2):
    ids, labels, mask = batch["input_ids"][row], batch["labels"][row], batch["attention_mask"][row]
    trained = labels != -100
    print(f"row {row}: real={int(mask.sum())} padded={int((mask == 0).sum())} trained={int(trained.sum())}")
    print("   trained text:", repr(tok.decode(ids[trained])))
# END SOLUTION

# %% [markdown]
# ## 5 · After: same tickets; compare validity, accuracy, latency (5')

# %%
after = evaluate(model, eval_df)
table = pd.DataFrame({"before": summarize(before), "after": summarize(after)}).T
print("All eval tickets")
print(table.round(3).to_string())
print("\nBy language")
pd.concat({"before": before, "after": after}, names=["model"]).groupby(["model", "lang"]).apply(
    lambda g: pd.Series(summarize(g))).round(3)

# %% [markdown]
# ✏️ **Exercise:** show up to 5 tickets where the **intent was wrong before and right after** (text, raw output before, raw output after). Then show any tickets that are still wrong after training. What kind of errors remain?

# %%
# BEGIN SOLUTION
cmp = before[["text", "intent", "raw", "pred_intent"]].join(after[["raw", "pred_intent"]], rsuffix="_after")
fixed = cmp[(cmp.pred_intent != cmp.intent) & (cmp.pred_intent_after == cmp.intent)]
print(f"fixed: {len(fixed)}")
print(fixed[["text", "raw", "raw_after"]].head(5).to_string())
print("\nstill wrong:")
print(cmp[cmp.pred_intent_after != cmp.intent][["text", "intent", "raw_after"]].head(5).to_string())
# END SOLUTION

# %% [markdown]
# 🔍 **Look at:** before tuning, outputs often parse but add extra keys ("reason", "priority2") and invent labels ("register", "CANCELLATION"). After LoRA, outputs follow the exact schema with valid label names, and intent accuracy rises sharply on English. Arabic improves less: every training example was English. Priority is a business rule (base level per intent + urgency words), so the model has to learn the rule from examples; it gets the common cases and misses rare "urgent" ones. (In quick mode the adapter has seen only a few dozen examples, so the change is small.)
#
# Save the adapter (~35 MB in fp32, not the ~2 GB base model) and the predictions for the leaderboard.

# %%
if ADAPTER_FALLBACK is None:
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    size_mb = sum(f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"saved adapter to {ADAPTER_DIR.relative_to(U.LABS)} ({size_mb:.0f} MB incl. tokenizer)")

for name, res in [("LLM LoRA (0.5B) before", before), ("LLM LoRA (0.5B) after", after)]:
    U.save_predictions(
        name,
        res.assign(y_true=res.intent, y_pred=res.pred_intent, confidence=np.nan)[
            ["ticket_id", "lang", "y_true", "y_pred", "confidence"]],
        {"day": 3, "p95_ms": float(np.percentile(res.ms, 95)), "device": DEV,
         "notes": f"{BASE} {'+ LoRA r=16' if 'after' in name else 'zero-shot'}; "
                  f"{len(records)} SFT records; JSON validity {res.valid.mean():.2f}; quick={U.QUICK}"})

# %% [markdown]
# ## ✱ Stretch · Merge the adapter; compare r = 4 vs. r = 32
#
# At inference, the adapter adds a small extra matmul per layer. `merge_and_unload()` folds `B·A` into the base weights, giving a plain model with zero adapter overhead (but you lose the ability to swap adapters).

# %%
sample = eval_df.head(U.pick(10, 2))
t_adapter = evaluate(model, sample).ms.mean()
merged = model.merge_and_unload()
res_merged = evaluate(merged, sample)
print(f"adapter: {t_adapter:.0f} ms/ticket | merged: {res_merged.ms.mean():.0f} ms/ticket | "
      f"same answers: {(res_merged.raw.values == after.head(len(sample)).raw.values).mean():.0%}")

# %% [markdown]
# **Rank experiment (try at home, ~2 × 8 min on a T4):** re-run sections 3–5 with `r=4, lora_alpha=8` and with `r=32, lora_alpha=64` (reload `base_model` first, since `get_peft_model` modifies it in place). Compare trainable parameters, adapter size, validity and intent accuracy. On a narrow format task like ours, r=4 is usually almost as good as r=32: rank is capacity you only need for harder behaviour changes.
#
# ```python
# base_model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(DEV)
# model = get_peft_model(base_model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
#                                               target_modules=lora.target_modules, task_type="CAUSAL_LM"))
# ```

# %% [markdown]
# ### Optional · QLoRA (GPU + bitsandbytes only)
#
# QLoRA loads the frozen base in 4-bit NF4 and trains the adapters in 16-bit on top. For a 0.5B model it isn't needed; for 7B models it is what makes a single 16 GB GPU enough. This cell only loads the 4-bit model and attaches adapters, to show the memory difference.

# %%
try:
    import bitsandbytes  # noqa: F401
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

if torch.cuda.is_available() and HAS_BNB:
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    q_base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0})
    q_base = prepare_model_for_kbit_training(q_base)
    q_model = get_peft_model(q_base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                target_modules=lora.target_modules, task_type="CAUSAL_LM"))
    q_model.print_trainable_parameters()
    print(f"4-bit base: {q_base.get_memory_footprint() / 1e9:.2f} GB vs fp32 merged: {merged.get_memory_footprint() / 1e9:.2f} GB")
else:
    print("Skipping QLoRA: it needs a CUDA GPU and the bitsandbytes package.")

# %% [markdown]
# ## Checkpoint
#
# 1. Why are prompt tokens labelled `-100`, and what would happen to the loss if they weren't?
# 2. LoRA with r=16 trains <2% of the parameters. Where does the memory saving come from?
# 3. The refund policy changes every quarter. Should it go into the LoRA training data?
#
# <details><summary>Answers</summary>
#
# 1. `-100` is ignored by the cross-entropy loss. Without masking, most of the loss would come from reproducing the (long, identical) system prompt and the ticket, diluting the signal from the answer and teaching the model to echo inputs.
# 2. Gradients and Adam states (≈ 12 of the ~16 bytes per parameter) are only kept for the adapter parameters; the frozen base needs just its weights (and in QLoRA, those are 4-bit).
# 3. No. Facts that change belong in RAG (Day 2). Fine-tune for format and behaviour; retrieve for knowledge.
#
# </details>
#
# ## Carry forward
#
# Lab 3.4 puts "LLM LoRA (0.5B) before/after" on the leaderboard next to the encoder. For your capstone: if you need structured LLM output, this notebook is the template (fixed prompt + SFT records + masking + LoRA + before/after table); if you don't fine-tune, the before/after table is still how you justify that decision.
