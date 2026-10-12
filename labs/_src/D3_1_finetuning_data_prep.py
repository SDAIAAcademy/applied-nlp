# %% [markdown]
# # Lab 3.1 · Prepare a leak-free ticket dataset
#
# Day 3 · Session 1 (Fine-tuning fundamentals & data) · **20 min**
#
# **What you'll build**
# - A check for near-duplicate tickets between train and test (the classic way templated data inflates scores).
# - A realistic low-resource training set (≤ 80 tickets per intent) next to the full one.
# - A tokenized Hugging Face `DatasetDict` + label maps saved to `artifacts/`. Labs 3.2 and 3.4 load them.
#
# **Tasks**
# - [ ] 1 · Explore label distribution and text lengths (3')
# - [ ] 2 · Near-duplicate check between train and test with embeddings (5')
# - [ ] 3 · Build a realistic low-resource train set (≤ 80 per intent) (3')
# - [ ] 4 · Label maps, DatasetDict, tokenization, save to disk (6')
# - [ ] 5 · Start the model download for Lab 3.2 / 3.3 in the background (3')
# - [ ] ✱ Generate 100 synthetic Arabic training tickets with the LLM; spot-check 10

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
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

pd.set_option("display.max_colwidth", 90)

train_df = U.load_tickets("train")
val_df = U.load_tickets("val")
test_df = U.load_tickets("test")
ar_df = U.load_tickets("ar_test")
LABELS = U.load_labels()
print({"train": len(train_df), "val": len(val_df), "test": len(test_df), "ar_test": len(ar_df)})

# %% [markdown]
# ## 1 · Explore label distribution and text lengths (3')
#
# Before training anything, look at the labels and at the length of the texts **in tokens of the model you will fine-tune** (DistilmBERT). The label counts tell you whether accuracy is meaningful; the lengths tell you which `max_length` to use.

# %%
counts = train_df.intent.value_counts().sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(counts.index, counts.values, color="tab:blue")
ax.set_title("Training tickets per intent")
ax.set_xlabel("number of tickets")
ax.set_ylabel("intent")
fig.tight_layout()
plt.show()
print("priority:", train_df.priority.value_counts().to_dict())

# %% [markdown]
# Our course data is **perfectly balanced** (200 per intent) because we sampled it that way. Real ticket queues are not (see the slide on imbalance), and `priority` here is already skewed: "urgent" is rare. That is why we report **macro-F1**, which weights every class equally.

# %%
MODEL_NAME = U.MODELS["encoder"]  # distilbert/distilbert-base-multilingual-cased
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

len_en = np.array([len(ids) for ids in tok(train_df.text.tolist())["input_ids"]])
len_ar = np.array([len(ids) for ids in tok(ar_df.text.tolist())["input_ids"]])

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(len_en, bins=40, alpha=0.7, density=True, label="EN train")
ax.hist(len_ar, bins=20, alpha=0.7, density=True, label="AR test")
ax.set_title("Token lengths with the DistilmBERT tokenizer")
ax.set_xlabel("tokens per ticket (incl. [CLS]/[SEP])")
ax.set_ylabel("density")
ax.legend()
plt.show()

pct = pd.DataFrame({name: np.percentile(arr, [50, 95, 99, 100]) for name, arr in [("EN train", len_en), ("AR test", len_ar)]},
                   index=["p50", "p95", "p99", "max"])
print(pct.round(1))

# max_length = p99 of the combined lengths, rounded up to a multiple of 8 (friendlier for GPUs)
p99 = np.percentile(np.concatenate([len_en, len_ar]), 99)
MAX_LEN = int(min(128, 8 * np.ceil(p99 / 8)))
print("MAX_LEN =", MAX_LEN)

# %% [markdown]
# 🔍 **Look at:** tickets are short. Padding everything to 512 would waste >80% of the compute. Arabic tickets tokenize into more pieces per word than English in this vocabulary (Lab 1.1), so check that `MAX_LEN` covers them too.

# %% [markdown]
# ## 2 · Near-duplicate check between train and test (5')
#
# Bitext tickets are generated from templates, so many test tickets are near-copies of training tickets ("I need help to cancel order A-123" vs. "I need help canceling order B-456"). A model can score well on them by memorizing. To measure this we embed both splits with e5 (cached on disk, you computed the same embeddings on Day 1) and, for every test ticket, find its most similar training ticket.

# %%
# Quick mode embeds a subset to save CPU time; the full run embeds everything (cached after the first run).
train_sub = train_df if not U.QUICK else train_df.sample(1500, random_state=42)
test_sub = test_df if not U.QUICK else test_df.sample(300, random_state=42)
E_train = U.embed(train_sub.text, prefix="query: ", cache_key="train" if not U.QUICK else "d31_train_quick")
E_test = U.embed(test_sub.text, prefix="query: ", cache_key="test" if not U.QUICK else "d31_test_quick")
print(E_train.shape, E_test.shape)

# %% [markdown]
# ✏️ **Exercise:** the embeddings are L2-normalized, so cosine similarity is a dot product.
# 1. For every test ticket compute its **max cosine** to any training ticket (and which one).
# 2. Count how many test tickets are above **0.95**, and what fraction of the test set that is.
# 3. Show 5 examples (test text, closest train text, similarity, both intents).
# 4. Plot a histogram of the max similarities.

# %%
# BEGIN SOLUTION
sims = E_test @ E_train.T                         # (n_test, n_train)
best = sims.argmax(axis=1)
max_sim = sims.max(axis=1)
dup = max_sim > 0.95
print(f"near-duplicates (> 0.95): {dup.sum()} / {len(dup)} = {dup.mean():.1%}")
ex = pd.DataFrame({"test": test_sub.text.values, "closest_train": train_sub.text.values[best],
                   "sim": max_sim, "test_intent": test_sub.intent.values,
                   "train_intent": train_sub.intent.values[best]})
print(ex[dup].sort_values("sim", ascending=False).head(5).round(3).to_string())
plt.hist(max_sim, bins=40)
plt.title("Max cosine similarity of each test ticket to the training set")
plt.xlabel("max cosine to any train ticket"); plt.ylabel("test tickets"); plt.show()
# END SOLUTION

# %% [markdown]
# 🔍 **What does this mean for test scores?** A large share of our test tickets have a near-twin in train, almost always with the same intent. Any model that memorizes surface patterns will look great on them. So:
# - Treat our English test scores as **optimistic**. Real customer messages are messier.
# - The Arabic test set was written separately (not from the same templates). It is a more honest test of generalization, and you should expect lower scores there.
# - In a real project, report scores **with and without** the near-duplicates, or split by template/customer (group split) so related tickets stay on one side.

# %% [markdown]
# ## 3 · Build a realistic low-resource train set (≤ 80 per intent) (3')
#
# Real teams rarely start with 200 clean examples per class. We simulate a realistic labelling budget: at most 80 tickets per intent, **stratified** (every intent present) and **seeded** (reproducible). We keep the full 200/intent set too, so Lab 3.2 can compare.

# %%
N_PER_INTENT = 80
# shuffle once with a seed, then keep the first N of each intent (works even if a class has < N)
train_small_df = (train_df.sample(frac=1, random_state=42)
                  .groupby("intent").head(N_PER_INTENT)
                  .sort_values(["intent", "ticket_id"]).reset_index(drop=True))
print(len(train_small_df), "tickets |", train_small_df.intent.nunique(), "intents |",
      "per intent:", train_small_df.intent.value_counts().agg(["min", "max"]).to_dict())

# %% [markdown]
# ✏️ **Exercise:** write two quick sanity checks (use `assert`):
# 1. `train_small_df` has every intent and no intent has more than 80 tickets.
# 2. No `ticket_id` appears in more than one of train / val / test (exact leakage).

# %%
# BEGIN SOLUTION
per_intent = train_small_df.intent.value_counts()
assert set(per_intent.index) == set(LABELS["intents"]) and per_intent.max() <= N_PER_INTENT
ids = {"train": set(train_df.ticket_id), "val": set(val_df.ticket_id), "test": set(test_df.ticket_id)}
assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
print("checks passed")
# END SOLUTION

# %% [markdown]
# ## 4 · Label maps, DatasetDict, tokenization, save to disk (6')
#
# Models predict integers; we need a fixed mapping between intent names and ids. We sort the names so the mapping is deterministic, and we **save it next to the data and later next to the model**. A shuffled label map is a classic silent production bug: the model still runs, it just answers with the wrong names.

# %%
labels = sorted(train_df.intent.unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
assert labels == sorted(LABELS["intents"])

def to_ds(df):
    # keep ticket_id and lang for per-language evaluation; the Trainer drops unused columns
    return Dataset.from_pandas(
        df.assign(label=df.intent.map(label2id))[["ticket_id", "text", "label", "lang"]],
        preserve_index=False)

ds = DatasetDict(train=to_ds(train_df), train_small=to_ds(train_small_df), validation=to_ds(val_df),
                 test=to_ds(test_df), test_ar=to_ds(ar_df))
ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True)
ds

# %% [markdown]
# `DataCollatorWithPadding` pads **each batch** to its own longest item (dynamic padding), instead of padding the whole dataset to `MAX_LEN`.
#
# ✏️ **Exercise:** take 4 examples from `ds["train_small"]` with different lengths, keep only the model inputs (`input_ids`, `attention_mask`, `label`), run them through `collator`, and print the shape of `input_ids` and the padded `attention_mask`. Decode the shortest one to see the `[PAD]` tokens.

# %%
collator = DataCollatorWithPadding(tok)

# %%
# BEGIN SOLUTION
cols = ["input_ids", "attention_mask", "label"]
sample = ds["train_small"].select(range(4))
batch = collator([{c: ex[c] for c in cols} for ex in sample])
print({k: tuple(v.shape) for k, v in batch.items()})
print(batch["attention_mask"])
shortest = int(batch["attention_mask"].sum(1).argmin())
print(tok.decode(batch["input_ids"][shortest]))
# END SOLUTION

# %%
DS_DIR = U.ARTIFACTS / "intent_ds"
U.ARTIFACTS.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(DS_DIR))
label_info = {"label2id": label2id, "id2label": {str(i): l for i, l in id2label.items()},
              "max_length": MAX_LEN, "model_name": MODEL_NAME, "n_per_intent_small": N_PER_INTENT,
              "seed": 42}
(U.ARTIFACTS / "intent_labels.json").write_text(json.dumps(label_info, indent=2), encoding="utf-8")

reloaded = load_from_disk(str(DS_DIR))
print({k: len(v) for k, v in reloaded.items()})
print("saved:", DS_DIR.relative_to(U.LABS), "+ artifacts/intent_labels.json")

# %% [markdown]
# ## 5 · Start the model downloads for Labs 3.2 / 3.3 (3')
#
# DistilmBERT (~540 MB) and Qwen2.5-0.5B-Instruct (~1 GB) take a while on a busy network. We start the download in a **background thread** so you can keep working; the files land in the Hugging Face cache and Labs 3.2/3.3 load them from there.

# %%
from huggingface_hub import snapshot_download

DOWNLOADS = [U.MODELS["encoder"], U.MODELS["lora_base"]]
download_status = {}

def download_all():
    for repo in DOWNLOADS:
        try:
            snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
            download_status[repo] = "ok"
        except Exception as e:  # network errors should not kill the lab
            download_status[repo] = f"failed: {e.__class__.__name__}"

dl_thread = threading.Thread(target=download_all, daemon=True)
dl_thread.start()
print("download started in the background:", DOWNLOADS)

# %% [markdown]
# ## ✱ Stretch · Synthetic Arabic training tickets
#
# Our training data is English only. One cheap way to get Arabic training examples is to ask an LLM to write them, then **check them by hand** before using them. Rules: generate only for training (never for test), keep the intent label you asked for, and spot-check at least 10.
#
# On CPU the local 0.5B model writes weak Arabic; with the 1.5B model on a GPU, or an API model via `LLM_BASE_URL`, quality is much better.

# %%
llm = U.get_llm()

SYN_INTENTS = ["track_order", "get_refund", "cancel_order", "payment_issue", "recover_password"][:U.pick(5, 2)]
CALLS_PER_INTENT = U.pick(4, 1)
PER_CALL = U.pick(5, 2)          # 5 x 4 x 5 = 100 tickets on the full run; 4 in quick mode
ARABIC = re.compile(r"[؀-ۿ]")
STYLES = ["Gulf dialect, informal", "Modern Standard Arabic, polite", "short and angry", "with a typo or two"]

rows = []
for intent in SYN_INTENTS:
    examples = train_df[train_df.intent == intent].text.sample(3, random_state=0).tolist()
    for k in range(CALLS_PER_INTENT):
        msgs = [
            {"role": "system", "content": "You write realistic customer support messages in Arabic for an online store."},
            {"role": "user", "content": (
                f"Write {PER_CALL} different Arabic customer messages with the intent '{intent}'. "
                f"Style: {STYLES[k % len(STYLES)]}. English examples of this intent:\n- " + "\n- ".join(examples) +
                f"\nReturn one message per line, no numbering, no translation.")},
        ]
        out = llm.chat(msgs, temperature=0.8, max_new_tokens=60 * PER_CALL)
        lines = [re.sub(r"^\s*(\d+[.)-]|[-*•])\s*", "", l).strip() for l in out.splitlines()]
        lines = [l for l in lines if ARABIC.search(l) and 5 <= len(l) <= 300]
        rows += [{"text": l, "intent": intent} for l in lines[:PER_CALL]]

syn = pd.DataFrame(rows).drop_duplicates("text")
if syn.empty:
    syn = pd.DataFrame(columns=["text", "intent"])
syn["category"] = syn.intent.map(LABELS["intent_to_category"])
syn["lang"] = "ar"
syn["source"] = f"synthetic:{llm.model}"
print(len(syn), "synthetic tickets")
syn.sample(min(10, len(syn)), random_state=0)

# %% [markdown]
# 🔍 **Spot-check:** read the 10 rows above. For each: is it Arabic? Does it match the intent? Is it realistic? A common failure is the model drifting to another intent or writing English. Drop bad rows before saving (here we save everything with a `source` column so you can filter later).

# %%
syn.to_csv(U.ARTIFACTS / "synthetic_ar.csv", index=False, encoding="utf-8")
print("saved artifacts/synthetic_ar.csv")

# %% [markdown]
# Make sure the background downloads finished before you move on:

# %%
dl_thread.join(timeout=U.pick(1800, 600))
print(download_status or "still downloading... re-run this cell")

# %% [markdown]
# ## Checkpoint
#
# 1. Why is a high test score on our English set weaker evidence than a high score on the Arabic set?
# 2. Why do we save `label2id`/`id2label` to disk instead of rebuilding them in each notebook?
# 3. You pick `max_length` = 512 "to be safe". What does it cost, and what does p99 give you instead?
#
# <details><summary>Answers</summary>
#
# 1. Many English test tickets are near-duplicates of training tickets (same templates), so the model can score by memorizing. The Arabic set was written separately.
# 2. If the order changes (a new label, a different sort), ids silently map to the wrong names. Saving the map with the data and model keeps them consistent.
# 3. Much slower training and inference (attention cost grows with length) for no gain; p99 covers 99% of tickets with a small truncation risk on the rest.
#
# </details>
#
# ## Carry forward
#
# Lab 3.2 loads `artifacts/intent_ds` and `artifacts/intent_labels.json` to fine-tune DistilmBERT; Lab 3.4 reuses the label maps. In your capstone, keep the same discipline: one saved, versioned dataset with fixed splits and label maps.
