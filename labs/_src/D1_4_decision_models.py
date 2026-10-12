# %% [markdown]
# # Lab 1.4 · Ticket classification + Day 1 integration
#
# Day 1 · Session 4 (Decision models) · 30 min. What you'll build:
# - three rungs of the classification ladder: TF-IDF + logistic regression, zero-shot NLI, e5 embeddings + logistic regression;
# - a proper evaluation (macro-F1, confusion matrix, English vs. Arabic slice) and a confidence threshold chosen on validation data;
# - `process_ticket(text)`: intent, confidence, route, category and the 3 most similar resolved tickets, for English and Arabic.
#
# **Tasks**
# - [ ] 1 · TF-IDF + logistic regression baseline (5')
# - [ ] 2 · Zero-shot NLI on categories; try two label wordings (7')
# - [ ] 3 · Embeddings + logistic regression on intents (5')
# - [ ] 4 · Report, confusion matrix, EN vs. AR slice (5')
# - [ ] 5 · Coverage–accuracy curve; choose τ on validation (4')
# - [ ] 6 · Integrate: `process_ticket()` = classify + top-3 similar (4')
# - [ ] ✱ Stretch · kNN-vote classifier from the search index; compare errors

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
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from transformers import pipeline

pd.set_option("display.max_colwidth", 100)

train, val, test = U.load_tickets("train"), U.load_tickets("val"), U.load_tickets("test")
ar = U.load_tickets("ar_test")
history = U.load_tickets("history")              # sorted by intent: subsample randomly, never with .head()
if U.QUICK:
    history = history.sample(1000, random_state=0).sort_index().reset_index(drop=True)
labels = U.load_labels()
print(f"train {len(train)} | val {len(val)} | test {len(test)} | AR test {len(ar)} | history {len(history)}")
print(f"{len(labels['intents'])} intents, {len(labels['categories'])} categories")

def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "macro-F1": f1_score(y_true, y_pred, average="macro")}

# %% [markdown]
# ## 1 · TF-IDF + logistic regression baseline (5')
#
# Always start with a cheap baseline: word and word-pair counts (TF-IDF) fed to a linear model.
# It trains in seconds on CPU and tells you how much a fancier model really adds.

# %%
tfidf_clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=10.0),
)
t0 = time.perf_counter()
tfidf_clf.fit(train.text, train.intent)
print(f"trained in {time.perf_counter() - t0:.1f}s on {len(train)} tickets, vocabulary: {len(tfidf_clf[0].vocabulary_):,} terms")

tfidf_pred_en = tfidf_clf.predict(test.text)
pd.DataFrame([scores(test.intent, tfidf_pred_en)], index=["TF-IDF + LogReg, EN test"]).round(3)

# %% [markdown]
# Bitext tickets are templated, so even this baseline is strong on English. Keep the number in mind.
#
# ## 2 · Zero-shot NLI on categories; try two label wordings (7')
#
# No training data at all: an NLI model judges whether the ticket (premise) entails "This customer message is about {label}."
# (hypothesis), once per label. First, the example from the slide:

# %%
import torch

# This checkpoint is stored in float16. transformers v5 keeps the stored dtype by default, and float16 is very slow on CPU,
# so we ask for float32 on CPU (float16 is fine on a GPU).
nli = pipeline("zero-shot-classification", model=U.MODELS["nli"], device=U.device(),
               dtype=torch.float16 if U.device() == "cuda" else torch.float32)

slide_labels = ["refund", "delivery", "payment problem", "account access", "order changes", "invoice"]
res = nli("لقد تم خصم المبلغ مرتين من بطاقتي", candidate_labels=slide_labels,
          hypothesis_template="This customer message is about {}.")
pd.Series(res["scores"], index=res["labels"]).round(3)

# %% [markdown]
# Now the 11 categories, on a sample of English test tickets and the Arabic tickets. Cost grows with the number of
# labels (11 forward passes per ticket), so we keep the sample small. Two wordings of the same labels:
# the raw names (`PAYMENT`, `CANCEL`, ...) and short descriptive phrases.

# %%
CATEGORIES = labels["categories"]
DESCRIPTIONS = {
    "ACCOUNT": "creating, editing, switching or deleting an account, or a password",
    "CANCEL": "the fee for cancelling",
    "CONTACT": "talking to customer service or a human agent",
    "DELIVERY": "delivery options and delivery times",
    "FEEDBACK": "a complaint or a review",
    "INVOICE": "an invoice or a bill",
    "ORDER": "placing, changing, cancelling or tracking an order",
    "PAYMENT": "payment methods or a payment problem",
    "REFUND": "getting a refund or the refund policy",
    "SHIPPING": "the shipping address",
    "SUBSCRIPTION": "the newsletter subscription",
}
desc_to_cat = {v: k for k, v in DESCRIPTIONS.items()}

zs_en = test.sample(U.pick(150, 20), random_state=0)
zs_ar = ar.sample(U.pick(len(ar), 20), random_state=0)

def zero_shot(texts, candidate_labels):
    outs = nli(list(texts), candidate_labels=candidate_labels, batch_size=16,
               hypothesis_template="This customer message is about {}.")
    return [o["labels"][0] for o in outs]

rows, zs_preds = [], {}
t0 = time.perf_counter()
for lang, df in [("EN", zs_en), ("AR", zs_ar)]:
    raw = zero_shot(df.text, CATEGORIES)
    desc = [desc_to_cat[p] for p in zero_shot(df.text, list(DESCRIPTIONS.values()))]
    zs_preds[lang] = pd.DataFrame({"text": df.text.to_numpy(), "category": df.category.to_numpy(), "raw": raw, "descriptive": desc})
    rows.append({"lang": lang, "n": len(df), "raw names": accuracy_score(df.category, raw),
                 "descriptive phrases": accuracy_score(df.category, desc)})
print(f"{sum(len(d) for d in zs_preds.values()) * 2 * len(CATEGORIES)} NLI passes in {time.perf_counter() - t0:.0f}s")
pd.DataFrame(rows).set_index("lang").round(3)

# %%
zs_preds["EN"].query("raw != category or descriptive != category").head(8)

# %% [markdown]
# 🔍 **Look at** how much the wording matters: same model, same tickets, accuracy moves by 15–20 points, and not necessarily
# in the same direction for English and Arabic. Read the errors: a phrase like "a complaint or a review" attracts every ticket
# that *reports* a problem, while a raw name like `CANCEL` (here it only means "cancellation fee") says little to the model.
# Label wording is a hyper-parameter: tune it on validation data. The samples are small (20 per language in quick mode),
# so treat the differences as indicative. Even the slide example is a close call between "refund" and "payment problem".
#
# Zero-shot is useful for a cold start or for labels that change, but it is clearly below a trained classifier, and every
# extra label costs another forward pass. With a few hundred labelled tickets, train something.
#
# ## 3 · Embeddings + logistic regression on intents (5')
#
# The best value-for-effort classifier: a frozen multilingual encoder turns each ticket into a 384-d vector once,
# then a linear model trains in seconds. Embeddings are cached in `labs/cache/` (shared with Lab 1.2).

# %%
t0 = time.perf_counter()
X_tr = U.embed(train.text, prefix="query: ", cache_key="train")
X_val = U.embed(val.text, prefix="query: ", cache_key="val")
X_te = U.embed(test.text, prefix="query: ", cache_key="test")
X_ar = U.embed(ar.text, prefix="query: ", cache_key="ar_test")
print(f"embeddings ready in {time.perf_counter() - t0:.0f}s:", X_tr.shape, X_val.shape, X_te.shape, X_ar.shape)

clf = LogisticRegression(max_iter=2000, C=4.0)
t0 = time.perf_counter()
clf.fit(X_tr, train.intent)
print(f"trained in {time.perf_counter() - t0:.1f}s")

P_te, P_ar = clf.predict_proba(X_te), clf.predict_proba(X_ar)
e5_pred_en, e5_conf_en = clf.classes_[P_te.argmax(1)], P_te.max(1)
e5_pred_ar, e5_conf_ar = clf.classes_[P_ar.argmax(1)], P_ar.max(1)
pd.DataFrame([scores(test.intent, e5_pred_en)], index=["e5 + LogReg, EN test"]).round(3)

# %% [markdown]
# ## 4 · Report, confusion matrix, EN vs. AR slice (5')
#
# Per-class precision, recall and F1 show which intents are weak; the confusion matrix shows what they are confused with.

# %%
print(classification_report(test.intent, e5_pred_en, digits=3))

# %%
U.plot_confusion(test.intent, e5_pred_en, labels=labels["intents"], title="e5 + LogReg, EN test (row-normalized)")
plt.show()
U.top_confusions(test.intent, e5_pred_en, k=8)

# %% [markdown]
# Now slice by language. We collect the predictions of both trained models on the English and the Arabic test sets
# in one long DataFrame.

# %%
tfidf_proba_en, tfidf_proba_ar = tfidf_clf.predict_proba(test.text), tfidf_clf.predict_proba(ar.text)
tfidf_classes = tfidf_clf.classes_

def pred_frame(df, y_pred, conf, model):
    return pd.DataFrame({"model": model, "ticket_id": df.ticket_id.to_numpy(), "lang": df.lang.to_numpy(),
                         "y_true": df.intent.to_numpy(), "y_pred": y_pred, "confidence": conf})

all_preds = pd.concat([
    pred_frame(test, tfidf_classes[tfidf_proba_en.argmax(1)], tfidf_proba_en.max(1), "TF-IDF + LogReg"),
    pred_frame(ar, tfidf_classes[tfidf_proba_ar.argmax(1)], tfidf_proba_ar.max(1), "TF-IDF + LogReg"),
    pred_frame(test, e5_pred_en, e5_conf_en, "e5 + LogReg"),
    pred_frame(ar, e5_pred_ar, e5_conf_ar, "e5 + LogReg"),
], ignore_index=True)
all_preds.sample(5, random_state=0)

# %% [markdown]
# ✏️ **Exercise** · Complete `slice_scores(df)`: one row per (`model`, `lang`) group with `n`, `accuracy` and `macro-F1`
# (use `f1_score(..., average="macro")`).

# %%
def slice_scores(df, by=("model", "lang")):
    rows = []
    for keys, g in df.groupby(list(by)):
        # BEGIN SOLUTION
        rows.append({**dict(zip(by, keys)), "n": len(g),
                     "accuracy": accuracy_score(g.y_true, g.y_pred),
                     "macro-F1": f1_score(g.y_true, g.y_pred, average="macro")})
        # END SOLUTION
    return pd.DataFrame(rows).set_index(list(by))

# %%
# Check + safety net
try:
    slice_table = slice_scores(all_preds)
except NotImplementedError:
    print("Exercise skipped: using the reference slice_scores()")
    def slice_scores(df, by=("model", "lang")):
        return pd.DataFrame([{**dict(zip(by, keys)), "n": len(g), "accuracy": accuracy_score(g.y_true, g.y_pred),
                              "macro-F1": f1_score(g.y_true, g.y_pred, average="macro")}
                             for keys, g in df.groupby(list(by))]).set_index(list(by))
    slice_table = slice_scores(all_preds)
slice_table.round(3)

# %% [markdown]
# Why does TF-IDF collapse on Arabic? Its vocabulary was learned from English training tickets, so almost every Arabic word
# is unknown and the ticket becomes an (almost) empty vector. The model then falls back on its intercepts and predicts
# nearly the same intent for everything. The multilingual encoder maps Arabic into the same space as English, so the
# classifier trained only on English still works: zero-shot cross-lingual transfer.

# %%
X_ar_tfidf = tfidf_clf[0].transform(ar.text)
print(f"Arabic tickets with an all-zero TF-IDF vector: {(X_ar_tfidf.sum(axis=1) == 0).mean():.0%}")
print("distinct intents predicted on Arabic, TF-IDF:", all_preds.query("model == 'TF-IDF + LogReg' and lang == 'ar'").y_pred.nunique(),
      "| e5:", len(set(e5_pred_ar)), "| true:", ar.intent.nunique())

ar_e5 = all_preds.query("model == 'e5 + LogReg' and lang == 'ar'").assign(dialect=ar.dialect.to_numpy())
slice_scores(ar_e5, by=("model", "dialect")).round(3)

# %% [markdown]
# ## 5 · Coverage–accuracy curve; choose τ on validation (4')
#
# Let the model abstain: if its confidence (max probability) is below τ, send the ticket to a human.
# Raising τ increases accuracy on what is auto-handled and lowers coverage. We choose τ on the **validation** set,
# then report what happens on test.

# %%
P_val = clf.predict_proba(X_val)
conf_val = P_val.max(1)
correct_val = clf.classes_[P_val.argmax(1)] == val.intent.to_numpy()
taus = np.linspace(0, 0.99, 100)

def coverage_accuracy(conf, correct, taus=taus):
    cov = np.array([(conf >= t).mean() for t in taus])
    acc = np.array([correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in taus])
    return cov, acc

cov_val, acc_val = coverage_accuracy(conf_val, correct_val)
print(f"validation accuracy without abstention: {correct_val.mean():.3f}")

# %% [markdown]
# On templated Bitext data the validation accuracy is already above 99% without abstaining, so a 95% target would give τ = 0
# (accept everything). We therefore use a strict target, 99.9%. With real tickets you would start around 90–95%.
#
# ✏️ **Exercise** · Complete `choose_tau(conf, correct, target)`: return the **smallest** τ in `taus` whose accuracy on the
# accepted tickets (`conf >= τ`) reaches `target`. The smallest such τ keeps the most coverage.

# %%
TARGET = 0.999   # at most 1 error per 1,000 auto-handled tickets

def choose_tau(conf, correct, target=TARGET, taus=taus):
    # BEGIN SOLUTION
    for t in taus:
        keep = conf >= t
        if keep.any() and correct[keep].mean() >= target:
            return float(t)
    return float(taus[-1])
    # END SOLUTION

# %%
# Check + safety net
try:
    TAU = choose_tau(conf_val, correct_val)
except NotImplementedError:
    print("Exercise skipped: using the reference choose_tau()")
    def choose_tau(conf, correct, target=TARGET, taus=taus):
        ok = [t for t in taus if (conf >= t).any() and correct[conf >= t].mean() >= target]
        return float(ok[0]) if ok else float(taus[-1])
    TAU = choose_tau(conf_val, correct_val)
print(f"τ = {TAU:.2f} for target accuracy {TARGET:.1%} on validation")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(taus, acc_val, label="accuracy on accepted (val)")
ax.plot(taus, cov_val, label="coverage (val)")
cov_ar_curve, acc_ar_curve = coverage_accuracy(e5_conf_ar, e5_pred_ar == ar.intent.to_numpy())
ax.plot(taus, acc_ar_curve, ls="--", label="accuracy on accepted (AR test)")
ax.plot(taus, cov_ar_curve, ls="--", label="coverage (AR test)")
ax.axvline(TAU, color="black", lw=0.8, ls=":", label=f"τ = {TAU:.2f}")
ax.set_xlabel("confidence threshold τ")
ax.set_ylabel("share")
ax.set_title("Coverage vs. accuracy, e5 + LogReg")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
rows = []
for name, conf, correct in [("val (EN)", conf_val, correct_val),
                            ("test (EN)", e5_conf_en, e5_pred_en == test.intent.to_numpy()),
                            ("test (AR)", e5_conf_ar, e5_pred_ar == ar.intent.to_numpy())]:
    keep = conf >= TAU
    rows.append({"split": name, "coverage": keep.mean(), "accuracy on accepted": correct[keep].mean() if keep.any() else np.nan,
                 "accuracy without abstention": correct.mean()})
pd.DataFrame(rows).set_index("split").round(3)

# %% [markdown]
# 🔍 **Look at** the Arabic row: the threshold chosen on English validation data sends most Arabic tickets to humans,
# which is the safe behaviour, but the accuracy on the Arabic tickets it does accept is below the target.
# Confidence is not calibrated across languages: choose and check thresholds per slice. Logistic-regression probabilities
# over 27 classes are also spread out, so τ looks low; what matters is the accuracy it buys, not its value.
#
# ## 6 · Integrate: `process_ticket()` = classify + top-3 similar (4')
#
# One embedding pass, two features: the same query vector feeds the classifier and the search over resolved tickets
# (indexed with the `"passage: "` prefix in Lab 1.2, loaded here from the cache).

# %%
hist_emb = U.embed(history.text, prefix="passage: ", cache_key="history")
intent_to_category = labels["intent_to_category"]

def process_ticket(text, tau=TAU, k=3):
    q = U.embed([text], prefix="query: ")              # (1, 384), normalized
    p = clf.predict_proba(q)[0]
    i = int(p.argmax())
    intent, conf = clf.classes_[i], float(p[i])
    sims = hist_emb @ q[0]
    top = np.argsort(-sims)[:k]
    similar = [{"ticket_id": int(history.ticket_id[j]), "score": round(float(sims[j]), 3), "intent": history.intent[j],
                "text": history.text[j], "response": history.response[j][:150].replace("\n", " ") + "..."} for j in top]
    return {"intent": intent, "confidence": round(conf, 3), "route": "auto" if conf >= tau else "human_review",
            "category": intent_to_category[intent], "similar": similar}

demo = [
    "I was charged twice for my last order, please fix it",
    "how long does delivery to Riyadh usually take?",
    "لم يصلني المبلغ المسترد حتى الآن",
    "ابي اغير عنوان الشحن حق طلبي",
]
for text in demo:
    r = process_ticket(text)
    print(f"\n{text}\n  -> intent={r['intent']}  conf={r['confidence']}  route={r['route']}  category={r['category']}")
    for s in r["similar"]:
        print(f"     [{s['score']}] ({s['intent']}) {s['text']}\n         reply: {s['response'][:110]}")

# %% [markdown]
# 🔍 **Look at** the first ticket: it is natural English, but phrased unlike the templated Bitext training tickets, so the
# classifier is unsure and the threshold sends it to a human. That is abstention doing its job. The Arabic tickets also get low
# confidence, while their similar tickets can still be useful to the agent who handles them.
#
# This is the Day 1 deliverable: a ticket in English or Arabic goes in; intent, confidence, a routing decision, a category
# and similar past cases (with the replies agents gave) come out.
#
# ### Save predictions for the leaderboard
#
# Lab 3.4 compares every model of the week. We save both trained classifiers' predictions on the English and Arabic test sets,
# with the p95 latency of classifying **one** ticket (embedding included for e5).

# %%
bench_texts = test.text.tolist()[:U.pick(200, 30)]
lat_tfidf = U.bench(lambda t: tfidf_clf.predict_proba([t]), bench_texts)
lat_e5 = U.bench(lambda t: clf.predict_proba(U.embed([t], prefix="query: ")), bench_texts)
print("TF-IDF + LogReg:", {k: round(v, 2) for k, v in lat_tfidf.items()})
print("e5 + LogReg    :", {k: round(v, 2) for k, v in lat_e5.items()})

for name, lat, note in [
    ("TF-IDF + LogReg", lat_tfidf, "word 1-2gram TF-IDF + LogisticRegression(C=10), trained on EN train"),
    ("e5 + LogReg", lat_e5, f"multilingual-e5-small (frozen, 'query: ') + LogisticRegression(C=4), trained on EN train; tau={TAU:.2f} chosen on val"),
]:
    df = all_preds.query("model == @name").drop(columns="model")
    U.save_predictions(name, df, {"day": 1, "p95_ms": round(lat["p95_ms"], 2), "notes": note + f"; device={U.device()}"})

# %% [markdown]
# ## ✱ Stretch · kNN-vote classifier from the search index; compare errors
#
# No training at all: retrieve the 10 most similar resolved tickets and take a majority vote over their intents.
# It reuses the search index from Lab 1.2 and gives you explainable predictions ("8 of 10 similar tickets were `get_refund`").
#
# ✏️ **Exercise** · Complete `knn_vote`: for each row of neighbour indices, return the most common intent (`Counter.most_common`).

# %%
def knn_vote(q_emb, k=10):
    top = np.argsort(-(q_emb @ hist_emb.T), axis=1)[:, :k]     # (n_queries, k) neighbour indices
    neigh_intents = history.intent.to_numpy()[top]                  # (n_queries, k)
    # BEGIN SOLUTION
    return np.array([Counter(row).most_common(1)[0][0] for row in neigh_intents])
    # END SOLUTION

# %%
try:
    knn_en, knn_ar = knn_vote(X_te), knn_vote(X_ar)
except NotImplementedError:
    print("Exercise skipped: using the reference knn_vote()")
    def knn_vote(q_emb, k=10):
        top = np.argsort(-(q_emb @ hist_emb.T), axis=1)[:, :k]
        return np.array([Counter(row).most_common(1)[0][0] for row in history.intent.to_numpy()[top]])
    knn_en, knn_ar = knn_vote(X_te), knn_vote(X_ar)

knn_preds = pd.concat([pred_frame(test, knn_en, np.nan, "kNN vote (k=10)"), pred_frame(ar, knn_ar, np.nan, "kNN vote (k=10)")])
slice_scores(pd.concat([all_preds.query("model == 'e5 + LogReg'"), knn_preds])).round(3)

# %%
cmp = pd.DataFrame({"text": test.text, "true": test.intent, "LR": e5_pred_en, "kNN": knn_en})
print("LR right, kNN wrong:", ((cmp.LR == cmp.true) & (cmp.kNN != cmp.true)).sum(),
      "| kNN right, LR wrong:", ((cmp.kNN == cmp.true) & (cmp.LR != cmp.true)).sum(),
      "| both wrong:", ((cmp.kNN != cmp.true) & (cmp.LR != cmp.true)).sum())
cmp[cmp.LR != cmp.kNN].head(8)

# %% [markdown]
# kNN needs no training and updates instantly when new resolved tickets are indexed, but it depends on the index covering
# every intent well (in quick mode the history is small) and votes are blunt. The linear head learns a decision boundary
# from all training tickets. When both are wrong on the same tickets, look at the labels: these are often genuinely ambiguous.
#
# ## Checkpoint
#
# 1. TF-IDF + LR is close to e5 + LR on English but collapses on Arabic. Why, and what does that tell you about baselines?
# 2. Why must τ be chosen on validation data and not on test data?
# 3. Your manager wants 99% accuracy on auto-handled tickets. What do you report besides accuracy?
#
# <details><summary>Answers</summary>
#
# 1. TF-IDF features are words from the English training vocabulary; Arabic tickets become empty vectors. A baseline only
#    measures what it can see: always evaluate per slice (language, dialect), not just overall.
# 2. Choosing τ on test makes the test score optimistic (you tuned on it). Validation is for decisions, test for the final number.
# 3. Coverage: the share of tickets handled automatically at that threshold (and per language). 99% accuracy at 20% coverage
#    may not be worth it; the cost table decides.
#
# </details>
#
# ## Carry forward
#
# - `process_ticket()` is the Day 1 pipeline; Day 2 adds an LLM that drafts a grounded reply and a cascade "e5 + LR → LLM"
#   for the tickets this classifier sends to `human_review`.
# - Day 3 fine-tunes DistilmBERT on the same intents; Lab 3.4 puts it on the leaderboard next to the two models saved here.
