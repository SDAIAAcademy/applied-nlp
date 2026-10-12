# %% [markdown]
# # Lab 3.2 · Fine-tune a ticket intent classifier
#
# Day 3 · Session 2 (Fine-tune a decision model) · **25 min** · GPU runtime recommended (Colab T4)
#
# **What you'll build**
# - A multilingual DistilBERT ("DistilmBERT") fine-tuned on our 27 intents with the Hugging Face `Trainer`, early stopping and best-checkpoint selection.
# - An honest test: English and Arabic, compared with the Day 1 frozen-embeddings baseline (e5 + logistic regression).
# - A saved model folder (`artifacts/intent-distilmbert/`) that Lab 3.4 serves behind a demo.
#
# About 5 minutes of training on a T4. **Start training first, then read the code while it runs.**
#
# **Tasks**
# - [ ] 1 · Load the saved dataset; build the model with label maps (3')
# - [ ] 2 · Train with Trainer + early stopping (8')
# - [ ] 3 · Plot train/val loss and macro-F1 per epoch (3')
# - [ ] 4 · Test once: English and Arabic; compare to Day 1 embeddings + LR (6')
# - [ ] 5 · Save model + tokenizer + label maps; reload; predict (5')
# - [ ] ✱ Data-size ablation: 10 / 25 / 80 per class; plot F1 vs. data

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
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, pipeline)

pd.set_option("display.max_colwidth", 80)
MODEL_NAME = U.MODELS["encoder"]  # distilbert/distilbert-base-multilingual-cased
USE_FP16 = torch.cuda.is_available()  # fp16 only on a GPU; never on CPU
print("fp16:", USE_FP16)

# %% [markdown]
# ## 1 · Load the saved dataset; build the model with label maps (3')
#
# We load the `DatasetDict` and label maps saved in Lab 3.1. If you skipped Lab 3.1 (or are on a fresh Colab runtime), the next cell rebuilds them with the same logic: same seed, same ≤ 80 per intent sample, same `max_length` rule.

# %%
DS_DIR = U.ARTIFACTS / "intent_ds"
LABELS_FILE = U.ARTIFACTS / "intent_labels.json"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def build_intent_ds(n_per_intent=80, seed=42):
    """Same steps as Lab 3.1: label maps -> DatasetDict -> tokenize (max_length = p99, multiple of 8)."""
    frames = {"train": U.load_tickets("train"), "validation": U.load_tickets("val"),
              "test": U.load_tickets("test"), "test_ar": U.load_tickets("ar_test")}
    frames["train_small"] = (frames["train"].sample(frac=1, random_state=seed)
                             .groupby("intent").head(n_per_intent)
                             .sort_values(["intent", "ticket_id"]).reset_index(drop=True))
    labels = sorted(frames["train"].intent.unique())
    label2id = {l: i for i, l in enumerate(labels)}
    lens = [len(x) for x in tok(pd.concat([frames["train"].text, frames["test_ar"].text]).tolist())["input_ids"]]
    max_len = int(min(128, 8 * np.ceil(np.percentile(lens, 99) / 8)))
    ds = DatasetDict({k: Dataset.from_pandas(df.assign(label=df.intent.map(label2id))[["ticket_id", "text", "label", "lang"]],
                                             preserve_index=False)
                      for k, df in frames.items()})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)
    info = {"label2id": label2id, "id2label": {str(i): l for l, i in label2id.items()}, "max_length": max_len,
            "model_name": MODEL_NAME, "n_per_intent_small": n_per_intent, "seed": seed}
    U.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DS_DIR))
    LABELS_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return ds, info


if DS_DIR.exists() and LABELS_FILE.exists():
    ds = load_from_disk(str(DS_DIR))
    info = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    print("loaded artifacts/intent_ds")
else:
    ds, info = build_intent_ds()
    print("rebuilt artifacts/intent_ds")

label2id = info["label2id"]
id2label = {int(i): l for i, l in info["id2label"].items()}
MAX_LEN = info["max_length"]
print({k: len(v) for k, v in ds.items()}, "| MAX_LEN =", MAX_LEN)

# %% [markdown]
# **Which training set?** We train on `train_small` (80 per intent, 2,160 tickets) by default. It is closer to a real labelling budget and trains in ~2 minutes on a T4. Switch `TRAIN_SPLIT = "train"` (200 per intent) to see how much more data buys: usually a small gain on English (it's templated) and a bit more on Arabic, at 2.5× the training time.
#
# Quick mode (CPU smoke test) uses a tiny subset and caps the number of steps. Its scores are **not** representative.

# %%
TRAIN_SPLIT = "train_small"   # or "train" for the full 200 per intent

train_ds = ds[TRAIN_SPLIT]
val_ds = ds["validation"]
if U.QUICK:
    train_ds = train_ds.shuffle(seed=42).select(range(540))   # ~20 per intent
    val_ds = val_ds.shuffle(seed=42).select(range(216))
print("train:", len(train_ds), "| validation:", len(val_ds))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters | head:", model.classifier)

# %% [markdown]
# The load report says the pre-training head (`vocab_*`) is unused and the new classification head (`pre_classifier`, `classifier`) is newly initialized. That is expected: the head is the only part that starts from random.

# %% [markdown]
# ## 2 · Train with Trainer + early stopping (8')
#
# The recipe from the slide:
# - `compute_metrics` turns logits into accuracy and **macro-F1** on the validation set after every epoch.
# - `load_best_model_at_end` + `metric_for_best_model="macro_f1"` keep the best epoch, not the last.
# - `EarlyStoppingCallback(patience=2)` stops if validation macro-F1 doesn't improve for 2 evaluations.
# - `warmup_steps=0.1`: in transformers v5 a float < 1 means "10% of the steps" (the old `warmup_ratio` argument was removed).
# - `fp16` only when a GPU is available.

# %%
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}


collator = DataCollatorWithPadding(tok)
RUN_DIR = U.CACHE / "runs" / "intent"

args = TrainingArguments(
    output_dir=str(RUN_DIR), learning_rate=3e-5, num_train_epochs=4, warmup_steps=0.1,
    per_device_train_batch_size=U.pick(32, 16), per_device_eval_batch_size=64, weight_decay=0.01,
    fp16=USE_FP16, seed=42, max_steps=U.pick(-1, 102),  # quick: 3 short epochs on CPU
    eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="macro_f1", report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  processing_class=tok, data_collator=collator, compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
train_result = trainer.train()
print(train_result.metrics)

# %% [markdown]
# ## 3 · Plot train/val loss and macro-F1 per epoch (3')
#
# `trainer.state.log_history` is a list of dicts: training logs have `loss`, evaluation logs have `eval_loss` and `eval_macro_f1`.

# %%
hist = pd.DataFrame(trainer.state.log_history)
train_log = hist.dropna(subset=["loss"])[["epoch", "loss"]] if "loss" in hist else pd.DataFrame(columns=["epoch", "loss"])
eval_log = hist.dropna(subset=["eval_loss"])[["epoch", "eval_loss", "eval_macro_f1", "eval_accuracy"]]
print(eval_log.round(3).to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(train_log.epoch, train_log.loss, "o-", label="train")
ax1.plot(eval_log.epoch, eval_log.eval_loss, "o-", label="validation")
ax1.set_title("Loss per epoch"); ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy loss"); ax1.legend()
ax2.plot(eval_log.epoch, eval_log.eval_macro_f1, "o-", color="tab:green")
ax2.set_title("Validation macro-F1 per epoch"); ax2.set_xlabel("epoch"); ax2.set_ylabel("macro-F1")
fig.tight_layout()
plt.show()
print("best checkpoint:", trainer.state.best_model_checkpoint, "| best macro-F1:", trainer.state.best_metric)

# %% [markdown]
# 🔍 **Look at:** are both losses still falling (train longer), or does validation loss turn up while train loss keeps falling (overfitting: the best checkpoint saves you)? On our templated data, validation macro-F1 is usually high after 1–2 epochs.

# %% [markdown]
# ## 4 · Test once: English and Arabic; compare to Day 1 (6')
#
# Everything above was chosen on **validation**. Now we touch the test sets once. The Arabic test set is a **zero-shot cross-lingual** test: the model saw only English tickets. Expect a drop.

# %%
test_en = ds["test"] if not U.QUICK else ds["test"].shuffle(seed=42).select(range(270))
test_ar = ds["test_ar"]


def predict_split(split):
    out = trainer.predict(split)
    probs = torch.softmax(torch.tensor(out.predictions, dtype=torch.float32), dim=-1).numpy()
    return pd.DataFrame({"ticket_id": split["ticket_id"], "lang": split["lang"], "text": split["text"],
                         "y_true": [id2label[i] for i in split["label"]],
                         "y_pred": [id2label[i] for i in probs.argmax(-1)],
                         "confidence": probs.max(-1)})


preds_en, preds_ar = predict_split(test_en), predict_split(test_ar)
ft_scores = {lang: f1_score(p.y_true, p.y_pred, average="macro") for lang, p in [("EN", preds_en), ("AR", preds_ar)]}
print({k: round(v, 3) for k, v in ft_scores.items()}, "| n:", len(preds_en), len(preds_ar))
print(classification_report(preds_en.y_true, preds_en.y_pred, digits=3, zero_division=0)[-300:])

# %% [markdown]
# ✏️ **Exercise:** where does Arabic lose the most?
# 1. Compute per-intent F1 for EN and for AR (`classification_report(..., output_dict=True, zero_division=0)`).
# 2. Put them in one DataFrame with a `drop = EN − AR` column and show the 5 intents with the largest drop.

# %%
# BEGIN SOLUTION
rep_en = classification_report(preds_en.y_true, preds_en.y_pred, output_dict=True, zero_division=0)
rep_ar = classification_report(preds_ar.y_true, preds_ar.y_pred, output_dict=True, zero_division=0)
per_intent = pd.DataFrame({"f1_en": {k: rep_en[k]["f1-score"] for k in label2id if k in rep_en},
                           "f1_ar": {k: rep_ar[k]["f1-score"] for k in label2id if k in rep_ar}})
per_intent["drop"] = per_intent.f1_en - per_intent.f1_ar
print(per_intent.sort_values("drop", ascending=False).head(5).round(3))
# END SOLUTION

# %% [markdown]
# ✏️ **Exercise:** list the 5 **most confident mistakes** across both languages (text, true, predicted, confidence). Confident mistakes are the dangerous ones in production: a confidence threshold won't catch them.

# %%
# BEGIN SOLUTION
errors = pd.concat([preds_en, preds_ar]).query("y_true != y_pred")
print(errors.sort_values("confidence", ascending=False)[["lang", "text", "y_true", "y_pred", "confidence"]]
      .head(5).round(3).to_string(index=False))
# END SOLUTION

# %% [markdown]
# **Compare with Day 1.** Lab 1.4 saved predictions for "e5 + LogReg" (frozen multilingual embeddings + logistic regression). If that file exists we compare on the same tickets; otherwise we retrain that baseline here, on the **same training tickets** as the fine-tuned model, which is the fairer comparison anyway.

# %%
saved = U.load_predictions()
ours = pd.concat([preds_en, preds_ar])

if "e5 + LogReg" in saved:
    base = saved["e5 + LogReg"][0]
    base = base[base.ticket_id.isin(ours.ticket_id)]
    base_note = "Day 1 file (trained on the full train set)"
else:
    tr = train_ds.to_pandas()
    E_tr = U.embed(tr.text, cache_key=f"d32_{TRAIN_SPLIT}_{len(tr)}")
    lr = LogisticRegression(max_iter=2000, C=10).fit(E_tr, tr.label.map(id2label))
    E_te = U.embed(ours.text, cache_key=f"d32_test_{len(ours)}")
    base = ours[["ticket_id", "lang", "y_true"]].assign(y_pred=lr.predict(E_te))
    base_note = f"recomputed here on {TRAIN_SPLIT} ({len(tr)} tickets)"

common = ours.merge(base[["ticket_id", "lang", "y_pred"]], on=["ticket_id", "lang"], suffixes=("", "_base"))
rows = []
for lang, g in common.groupby("lang"):
    rows.append({"lang": lang, "n": len(g),
                 "e5 + LogReg": f1_score(g.y_true, g.y_pred_base, average="macro"),
                 "Fine-tuned DistilmBERT": f1_score(g.y_true, g.y_pred, average="macro")})
print("baseline:", base_note)
pd.DataFrame(rows).round(3)

# %% [markdown]
# 🔍 **Look at:** on English both approaches are high (templated data, near-duplicates). The interesting column is Arabic: e5 was trained with multilingual sentence-similarity objectives, so its frozen embeddings often transfer to Arabic *better* than a model fine-tuned only on English. Fine-tuning made the representations specific to English tickets. A few hundred real Arabic examples (or the synthetic ones from Lab 3.1, after review) usually close much of the gap.

# %% [markdown]
# ## 5 · Save model + tokenizer + label maps; reload; predict (5')
#
# A model folder is only useful if it contains everything needed to rebuild predictions: weights, tokenizer, and the label maps (stored in `config.json` as `id2label`/`label2id`; we also keep our JSON file next to it).

# %%
OUT = U.ARTIFACTS / "intent-distilmbert"
trainer.save_model(str(OUT))          # weights + config (with id2label / label2id)
tok.save_pretrained(str(OUT))
shutil.copy(LABELS_FILE, OUT / "intent_labels.json")
print(sorted(p.name for p in OUT.iterdir()))

clf = pipeline("text-classification", model=str(OUT), device=0 if torch.cuda.is_available() else -1,
               truncation=True, max_length=MAX_LEN)

# %% [markdown]
# ✏️ **Exercise:** use the reloaded `clf` to classify the examples below in one batched call (`batch_size=...`) and print a small table (text, label, score). Then show the **top 3** intents for the Arabic message "وين طلبي؟ صار له أسبوع" with `top_k=3`.

# %%
examples = ["where is my refund", "I was charged twice for order A-55821!!", "how do I delete my account?",
            "وين طلبي؟ صار له أسبوع", "أريد إلغاء طلبي رقم A-10023", "نسيت كلمة المرور"]

# %%
# BEGIN SOLUTION
out = clf(examples, batch_size=16)
print(pd.DataFrame({"text": examples, "label": [o["label"] for o in out],
                    "score": [round(o["score"], 3) for o in out]}).to_string(index=False))
print(clf("وين طلبي؟ صار له أسبوع", top_k=3))
# END SOLUTION

# %% [markdown]
# Save the test predictions for the Lab 3.4 leaderboard, with the single-ticket latency (p95) of the reloaded pipeline.

# %%
lat = U.bench(lambda t: clf(t), preds_en.text.tolist()[:U.pick(100, 20)])
print({k: round(v, 1) for k, v in lat.items()})
U.save_predictions(
    "Fine-tuned DistilmBERT",
    ours[["ticket_id", "lang", "y_true", "y_pred", "confidence"]],
    {"day": 3, "p95_ms": lat["p95_ms"], "device": U.device(),
     "notes": f"{MODEL_NAME} fine-tuned on {TRAIN_SPLIT} ({len(train_ds)} tickets); quick={U.QUICK}"})

# %% [markdown]
# ## ✱ Stretch · Data-size ablation: 10 / 25 / 80 per class
#
# How much labelled data do you need? Train fresh models on 10, 25 and 80 tickets per intent and plot validation macro-F1 against the data size. (We use validation, not test: this is a model-selection question.) Each run takes ~1 minute on a T4. Small sets need more epochs to converge, so we give every run 6 epochs.

# %%
def train_eval(n_per_class, seed=42):
    df = ds["train"].to_pandas()
    idx = df.sample(frac=1, random_state=seed).groupby("label").head(n_per_class).index.tolist()
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    a = TrainingArguments(output_dir=str(U.CACHE / "runs" / f"ablation-{n_per_class}"), learning_rate=5e-5,
                          num_train_epochs=6, warmup_steps=0.1, per_device_train_batch_size=U.pick(32, 16),
                          per_device_eval_batch_size=64, weight_decay=0.01, fp16=USE_FP16, seed=seed,
                          max_steps=U.pick(-1, 10), save_strategy="no", eval_strategy="no",
                          logging_strategy="no", report_to="none")
    t = Trainer(model=m, args=a, train_dataset=ds["train"].select(idx), eval_dataset=val_ds,
                processing_class=tok, data_collator=collator, compute_metrics=compute_metrics)
    t.train()
    return t.evaluate()["eval_macro_f1"]


sizes = [10, 25, 80]
ablation = pd.DataFrame({"per_class": sizes, "val_macro_f1": [train_eval(n) for n in sizes]})
print(ablation.round(3))
plt.plot(ablation.per_class, ablation.val_macro_f1, "o-")
plt.title("Validation macro-F1 vs. training data size")
plt.xlabel("training tickets per intent"); plt.ylabel("macro-F1"); plt.xscale("log")
plt.xticks(sizes, [str(s) for s in sizes])
plt.show()

# %% [markdown]
# ## Checkpoint
#
# 1. Validation macro-F1 peaked at epoch 2 and the run continued to epoch 4. Which weights does `trainer.model` hold after training, and why?
# 2. Why can the fine-tuned model lose to frozen e5 embeddings on Arabic, even though it wins on English?
# 3. You deploy the model but rebuild `id2label` with `set(train.intent)` in the service. What can go wrong?
#
# <details><summary>Answers</summary>
#
# 1. The epoch-2 checkpoint: `load_best_model_at_end=True` with `metric_for_best_model="macro_f1"` reloads the best evaluation checkpoint at the end.
# 2. Fine-tuning on English-only data specializes the representations to English tickets; e5's multilingual training kept EN and AR aligned. The cross-lingual signal can partly wash out.
# 3. Python sets have no stable order, so ids can map to the wrong intent names. Always load the saved label maps (they are in `config.json`).
#
# </details>
#
# ## Carry forward
#
# Lab 3.4 loads `artifacts/intent-distilmbert/` for behavioural tests, quantization and the Gradio demo, and puts "Fine-tuned DistilmBERT" on the leaderboard. For your capstone, this notebook is the template for any encoder classifier: change the dataset and label maps, keep the recipe.
