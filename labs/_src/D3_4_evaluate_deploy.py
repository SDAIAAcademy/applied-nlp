# %% [markdown]
# # Lab 3.4 · Leaderboard, error analysis, demo
#
# Day 3 · Session 4 (Evaluate, deploy & capstone kickoff) · **20 min** · CPU is fine
#
# **What you'll build**
# - One honest leaderboard of every intent model from the bootcamp, with bootstrap confidence intervals.
# - Structured error analysis and a behavioural test suite ("unit tests for models").
# - A fast CPU model (batching, int8 quantization) behind a Gradio demo of the full ticket assistant, plus an optional FastAPI endpoint.
#
# This notebook is also your **capstone template** for evaluation and serving.
#
# **Tasks**
# - [ ] 1 · Load saved predictions; build the leaderboard with bootstrap CIs (5')
# - [ ] 2 · Confusion pairs + tag 10 errors by cause (5')
# - [ ] 3 · Run the behavioural test suite (3')
# - [ ] 4 · Batch vs. single latency; CPU int8 quantization (3')
# - [ ] 5 · Launch the Gradio demo of the full assistant (4')
# - [ ] ✱ Wrap `assist()` in a FastAPI endpoint

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
import copy
import io
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from prepare_data import priority_for   # the business rule that produced the priority labels

pd.set_option("display.max_colwidth", 80)
train_df = U.load_tickets("train")
test_df = U.load_tickets("test")
ar_df = U.load_tickets("ar_test")
LABELS = U.load_labels()
texts_by_id = pd.concat([test_df, ar_df]).set_index("ticket_id").text

# %% [markdown]
# ## 1 · Load saved predictions; build the leaderboard with bootstrap CIs (5')
#
# Every lab that evaluated an intent classifier saved a prediction file in `artifacts/predictions/`. We load whatever exists. If the Day 1 TF-IDF baseline is missing (you skipped Lab 1.4), we compute it here so the table always has a baseline.

# %%
preds = U.load_predictions()
print("found:", list(preds) or "nothing yet")

if "TF-IDF + LogReg" not in preds:
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                          LogisticRegression(max_iter=2000, C=10)).fit(train_df.text, train_df.intent)
    both = pd.concat([test_df, ar_df], ignore_index=True)
    proba = tfidf.predict_proba(both.text)
    df = pd.DataFrame({"ticket_id": both.ticket_id, "lang": both.lang, "y_true": both.intent,
                       "y_pred": tfidf.classes_[proba.argmax(1)], "confidence": proba.max(1)})
    lat = U.bench(lambda t: tfidf.predict([t]), test_df.text.tolist()[:100])
    preds["TF-IDF + LogReg"] = (df, {"name": "TF-IDF + LogReg", "day": 1, "p95_ms": lat["p95_ms"],
                                     "notes": "computed inline in Lab 3.4"})
    print("TF-IDF baseline computed inline")

# %% [markdown]
# **Comparing fairly.** Not every model was evaluated on the same tickets: LLMs were run on a subset (e.g. 100–200 English tickets) because they are slow, and quick mode shrinks everything. A macro-F1 on 100 tickets has a much wider confidence interval than on 1,080. So the table shows `n` per language, a 95% bootstrap CI, and a column restricted to the English tickets that **every** model was evaluated on.
#
# Macro-F1 here averages over the intents present in the gold labels of each subset, so an invalid LLM output ("INVALID") counts as a miss.

# %%
def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)


N_BOOT = U.pick(1000, 200)
for name, (df, meta) in preds.items():
    df["y_pred"] = df.y_pred.fillna("INVALID").astype(str)

common_en = set.intersection(*[set(df[df.lang == "en"].ticket_id) for df, _ in preds.values()])
rows = []
for name, (df, meta) in preds.items():
    row = {"model": name, "day": meta.get("day")}
    for lang in ["en", "ar"]:
        g = df[df.lang == lang]
        row[f"n_{lang}"] = len(g)
        if len(g):
            f1, lo, hi = U.bootstrap_ci(g.y_true, g.y_pred, macro_f1, n=N_BOOT)
            row[f"F1_{lang}"], row[f"CI_{lang}"] = f1, f"[{lo:.3f}, {hi:.3f}]"
    g = df[(df.lang == "en") & df.ticket_id.isin(common_en)]
    row["F1_en_common"] = macro_f1(g.y_true, g.y_pred) if len(g) else np.nan
    row["p95_ms"] = meta.get("p95_ms")
    rows.append(row)

board = pd.DataFrame(rows).sort_values("F1_en", ascending=False).reset_index(drop=True)
print(f"common English tickets across all models: {len(common_en)}")
board.round(3)

# %%
fig, ax = plt.subplots(figsize=(9, 0.6 * len(board) + 1.5))
y = np.arange(len(board))
for k, (lang, color) in enumerate([("en", "tab:blue"), ("ar", "tab:orange")]):
    f1 = board[f"F1_{lang}"].values.astype(float)
    ci = board[f"CI_{lang}"].fillna("[nan, nan]").str.strip("[]").str.split(", ", expand=True).astype(float).values
    ax.errorbar(f1, y + (k - 0.5) * 0.25, xerr=[f1 - ci[:, 0], ci[:, 1] - f1], fmt="o", color=color,
                capsize=3, label=lang.upper())
ax.set_yticks(y, board.model)
ax.invert_yaxis()
ax.set_title("Intent macro-F1 with 95% bootstrap CIs")
ax.set_xlabel("macro-F1")
ax.set_ylabel("model")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at:** overlapping intervals mean "we can't tell these apart on this test set". English scores cluster near the top (templated data, near-duplicates from Lab 3.1). The Arabic column, with only 108 tickets, separates models that transfer across languages from those that don't, but with wide intervals. Latency differs by orders of magnitude: that often decides the winner.

# %% [markdown]
# ## 2 · Confusion pairs + tag 10 errors by cause (5')
#
# We pick the best model on English (among models evaluated on at least 100 English tickets, if any) and look at which intents it confuses.

# %%
eligible = board[board.n_en >= 100] if (board.n_en >= 100).any() else board
BEST = eligible.iloc[0].model
best_df = preds[BEST][0].copy()
best_df["text"] = best_df.ticket_id.map(texts_by_id)
print("best model:", BEST)
print(U.top_confusions(best_df.y_true, best_df.y_pred, k=8).to_string(index=False))

# %% [markdown]
# Now read the errors. We take 10 (up to 3 Arabic) into a table with an empty `cause` column.

# %%
errs = best_df[best_df.y_true != best_df.y_pred]
errors = pd.concat([errs[errs.lang == "ar"].head(3), errs[errs.lang == "en"]]).head(10).reset_index(drop=True)
errors = errors[["ticket_id", "lang", "text", "y_true", "y_pred", "confidence"]].assign(cause="")
CAUSES = ["label noise", "ambiguous", "coverage", "model error", "out of scope"]
errors.round(3)

# %% [markdown]
# ✏️ **Exercise:** read each error and fill the `cause` column with one of `CAUSES`:
# - **label noise**: the gold label is wrong,
# - **ambiguous**: a human could pick either label (e.g. `get_invoice` vs `check_invoice`),
# - **coverage**: a slice the training data doesn't cover (Arabic, dialect, typos),
# - **model error**: clear ticket, clear label, the model is just wrong,
# - **out of scope**: the ticket doesn't fit any intent.
#
# Then count the causes. What is the most useful next action? (See the slide's cause → action table.)
#
# (The reference answer uses a rough heuristic so it runs on any errors; in class, tag by reading.)

# %%
# BEGIN SOLUTION
NEAR_PAIRS = [{"get_invoice", "check_invoice"}, {"get_refund", "track_refund"}, {"get_refund", "check_refund_policy"},
              {"contact_customer_service", "contact_human_agent"}, {"delivery_options", "delivery_period"},
              {"change_shipping_address", "set_up_shipping_address"}, {"create_account", "registration_problems"},
              {"edit_account", "switch_account"}, {"cancel_order", "check_cancellation_fee"}]

def tag(r):
    if r.lang == "ar":
        return "coverage"
    if {r.y_true, r.y_pred} in NEAR_PAIRS or len(str(r.text).split()) <= 3:
        return "ambiguous"
    return "model error"

errors["cause"] = errors.apply(tag, axis=1)
print(errors[["text", "y_true", "y_pred", "cause"]].to_string())
print(errors.cause.value_counts())
# END SOLUTION

# %% [markdown]
# ## 3 · Run the behavioural test suite (3')
#
# Behavioural tests check **capabilities**, not average scores: things that must not change the label (invariance), changes that must move the output in a known direction (directional), and obvious cases every model must get right (minimum functionality).
#
# We test the fine-tuned model from Lab 3.2 if it exists, otherwise e5 + logistic regression trained here. We serve on **CPU**: that's where most ticket systems run, and it lets us quantize in section 4.

# %%
FT_DIR = U.ARTIFACTS / "intent-distilmbert"

if FT_DIR.exists():
    from transformers import pipeline

    MAX_LEN = json.loads((FT_DIR / "intent_labels.json").read_text(encoding="utf-8"))["max_length"]
    clf = pipeline("text-classification", model=str(FT_DIR), device=-1, truncation=True, max_length=MAX_LEN)
    MODEL_KIND = "Fine-tuned DistilmBERT"

    def predict_batch(texts, batch_size=32):
        out = clf(list(texts), batch_size=batch_size)
        return [o["label"] for o in out], [o["score"] for o in out]

    get_module = lambda: clf.model
    def set_module(m):
        clf.model = m
else:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(U.MODELS["embed"], device="cpu")
    fit_df = train_df if not U.QUICK else train_df.groupby("intent").head(40)
    lr = LogisticRegression(max_iter=2000, C=10).fit(U.embed(fit_df.text, cache_key=f"d34_train_{len(fit_df)}"), fit_df.intent)
    MODEL_KIND = "e5 + LogReg (trained here)"

    def predict_batch(texts, batch_size=32):
        E = st.encode(["query: " + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                      show_progress_bar=False)
        P = lr.predict_proba(E)
        return list(lr.classes_[P.argmax(1)]), list(P.max(1))

    get_module = lambda: st
    def set_module(m):
        global st
        st = m


def predict(text):
    return predict_batch([text])[0][0]


print(MODEL_KIND, "|", predict("I want to cancel order A-10001"))

# %%
TESTS = []  # (name, kind, fn)

def add_test(name, kind):
    def deco(fn):
        TESTS.append((name, kind, fn))
        return fn
    return deco


@add_test("order number A-10001 vs Z-99231", "invariance")
def _():
    a, b = predict("I want to cancel order A-10001"), predict("I want to cancel order Z-99231")
    assert a == b, f"{a} != {b}"

@add_test("customer name Sara vs Fahad", "invariance")
def _():
    a, b = predict("Hi, this is Sara, I can't log in to my account"), predict("Hi, this is Fahad, I can't log in to my account")
    assert a == b, f"{a} != {b}"

@add_test("typos", "invariance")
def _():
    a, b = predict("how can I track my order?"), predict("how cn i trak my ordr")
    assert a == b, f"{a} != {b}"

@add_test("polite vs rude", "invariance")
def _():
    a = predict("Could you please help me get a refund for my order? Thank you")
    b = predict("Give me a refund for my order. This is ridiculous")
    assert a == b, f"{a} != {b}"

PRIORITY_RANK = {p: i for i, p in enumerate(LABELS["priorities"])}

@add_test("urgency words never lower priority", "directional")
def _():
    for t in ["I want to cancel my order", "where is my invoice", "I was charged twice for my order"]:
        u = "URGENT!! " + t + " immediately"
        p0, p1 = priority_for(predict(t), t), priority_for(predict(u), u)
        assert PRIORITY_RANK[p1] >= PRIORITY_RANK[p0], f"{t!r}: {p0} -> {p1}"

MFT = [("cancel my order", "cancel_order"), ("where is my order", "track_order"),
       ("I forgot my password", "recover_password"), ("I want to delete my account", "delete_account"),
       ("I need a copy of my invoice", "get_invoice"), ("where is my refund", "track_refund"),
       ("I want to talk to a human agent", "contact_human_agent"), ("what payment methods do you accept", "check_payment_methods"),
       ("unsubscribe me from the newsletter", "newsletter_subscription"), ("I want to change my shipping address", "change_shipping_address")]
for text, gold in MFT:
    def _(text=text, gold=gold):
        got = predict(text)
        assert got == gold, f"got {got}"
    TESTS.append((f"'{text}' -> {gold}", "minimum functionality", _))

print(len(TESTS), "tests")

# %% [markdown]
# ✏️ **Exercise:** add an **EN ↔ AR invariance** test: "I want to cancel my order" and "أريد إلغاء طلبي" should get the same intent. Register it with `@add_test(..., "invariance")` so the runner below picks it up.

# %%
# BEGIN SOLUTION
@add_test("EN vs AR: cancel order", "invariance")
def _():
    a, b = predict("I want to cancel my order"), predict("أريد إلغاء طلبي")
    assert a == b, f"{a} != {b}"
# END SOLUTION

# %%
def run_tests(tests):
    rows = []
    for name, kind, fn in tests:
        try:
            fn()
            rows.append({"kind": kind, "test": name, "passed": True, "detail": ""})
        except AssertionError as e:
            rows.append({"kind": kind, "test": name, "passed": False, "detail": str(e)})
    return pd.DataFrame(rows)


results = run_tests(TESTS)
print(results.groupby("kind").passed.agg(["sum", "count"]).rename(columns={"sum": "passed", "count": "total"}))
results

# %% [markdown]
# 🔍 **Look at:** a model with 0.95+ macro-F1 can still fail typos or the Arabic version of a trivial request. Each failing test is a concrete, reproducible bug report. Keep this suite and re-run it on every new model (like unit tests in CI).

# %% [markdown]
# ## 4 · Batch vs. single latency; CPU int8 quantization (3')
#
# ✏️ **Exercise:** measure the time **per ticket** for `N` test tickets (a) one call per ticket and (b) one batched call (`predict_batch(texts, batch_size=32)`). Print both in ms/ticket and the speed-up.

# %%
N_LAT = U.pick(256, 64)
lat_texts = test_df.text.tolist()[:N_LAT]

# %%
# BEGIN SOLUTION
t0 = time.perf_counter()
for t in lat_texts:
    predict(t)
single_ms = (time.perf_counter() - t0) * 1000 / N_LAT
t0 = time.perf_counter()
predict_batch(lat_texts, batch_size=32)
batch_ms = (time.perf_counter() - t0) * 1000 / N_LAT
print(f"single: {single_ms:.1f} ms/ticket | batched: {batch_ms:.1f} ms/ticket | speed-up x{single_ms / batch_ms:.1f}")
# END SOLUTION

# %% [markdown]
# **Dynamic int8 quantization** stores the weights of every `nn.Linear` layer as 8-bit integers and quantizes activations on the fly. It is one line, CPU-only, and usually gives a 1.5–2× speed-up for a tiny accuracy loss. The `Linear` weights shrink 4×, but the large multilingual embedding matrix (119k tokens × 768) is not a `Linear` layer and stays fp32, so the whole model shrinks much less. Gains depend on the CPU (and on how busy it is): always **measure** both. (Recent PyTorch prints a notice that eager-mode quantization is moving to the `torchao` package; `quantize_dynamic` still works.)

# %%
from torch.ao.quantization import quantize_dynamic


def size_mb(module):
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell() / 1e6


def measure(label):
    sub = test_df.head(U.pick(500, 150))
    pred, _ = predict_batch(sub.text.tolist())
    lat = U.bench(predict, sub.text.tolist()[:U.pick(100, 30)])
    return {"model": label, "accuracy": float(np.mean(np.array(pred) == sub.intent.values)),
            "p50_ms": lat["p50_ms"], "p95_ms": lat["p95_ms"], "size_mb": size_mb(get_module())}


fp32_module = get_module()
quant = [measure(f"{MODEL_KIND} fp32")]
set_module(quantize_dynamic(copy.deepcopy(fp32_module), {torch.nn.Linear}, dtype=torch.qint8))
quant.append(measure(f"{MODEL_KIND} int8"))
set_module(fp32_module)   # keep fp32 for the demo; switch if the numbers convince you
pd.DataFrame(quant).round(3)

# %% [markdown]
# ## 5 · Launch the Gradio demo of the full assistant (4')
#
# `assist(ticket)` puts the week together:
# - **intent + confidence** from the classifier, **category** from the label map, **priority** from the business rule,
# - **top-3 similar resolved tickets** from the history (e5 embeddings, Day 1/2),
# - a **draft reply**: by default the agent reply of the most similar resolved ticket; with `USE_LLM = True`, the LLM writes a reply grounded in those similar replies (Day 2).

# %%
history = U.load_tickets("history")
if U.QUICK:
    history = history.sample(1000, random_state=42).reset_index(drop=True)
H = U.embed(history.text, prefix="query: ", cache_key="history" if not U.QUICK else "d34_history_quick")
USE_LLM = False   # True: draft the reply with U.get_llm() (slower; best on a GPU)


def assist(ticket, use_llm=None):
    use_llm = USE_LLM if use_llm is None else use_llm
    labels, scores = predict_batch([ticket])
    intent, conf = labels[0], float(scores[0])
    q = U.embed([ticket], prefix="query: ")[0]
    top = np.argsort(-(H @ q))[:3]
    similar = [{"ticket_id": int(history.ticket_id[i]), "text": history.text[i], "intent": history.intent[i],
                "similarity": round(float(H[i] @ q), 3)} for i in top]
    reply = history.response[top[0]]
    if use_llm:
        examples = "\n---\n".join(history.response[i] for i in top)
        reply = U.get_llm().chat([
            {"role": "system", "content": "You are a DemoMart support agent. Write a short, polite reply in the "
                                          "customer's language. Use only facts from the example replies."},
            {"role": "user", "content": f"Ticket: {ticket}\nIntent: {intent}\n\nExample replies:\n{examples}"}],
            max_new_tokens=200)
    return {"intent": intent, "confidence": round(conf, 3), "category": LABELS["intent_to_category"][intent],
            "priority": priority_for(intent, ticket), "needs_review": conf < 0.5,
            "reply": reply, "similar": similar}


for t in ["Where is my refund? It has been two weeks!!", "لم يصلني الطلب حتى الآن"]:
    out = assist(t)
    print(json.dumps({k: v for k, v in out.items() if k != "similar"}, ensure_ascii=False)[:400])
    print("  similar:", [s["text"] for s in out["similar"]])

# %%
# The LLM path: on CPU this loads Qwen2.5-0.5B, which writes fluent but often ungrounded steps.
# Read the draft critically: that is why USE_LLM is off by default and why drafts go to a human.
print(assist("I was charged twice for order A-55821", use_llm=True)["reply"][:600])

# %%
import gradio as gr


def run(ticket):
    out = assist(ticket)
    return out["intent"], out["confidence"], out["priority"], out["reply"], out


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=4, label="Customer message"),
    outputs=[gr.Textbox(label="Intent"), gr.Number(label="Confidence"), gr.Textbox(label="Priority"),
             gr.Textbox(label="Draft reply", lines=6), gr.JSON(label="Full output")],
    examples=[["Where is my refund?"], ["I want to cancel order A-10023 right now!!"],
              ["لم يصلني الطلب"], ["نسيت كلمة المرور وما أقدر أدخل حسابي"]],
    title="DemoMart ticket assistant",
)
if not U.QUICK:
    demo.launch(share=IN_COLAB)   # inline in Jupyter; a public link on Colab
else:
    print("quick mode: not launching the server. Direct call:", run("Where is my refund?")[:3])

# %% [markdown]
# ## ✱ Stretch · Wrap `assist()` in a FastAPI endpoint
#
# Gradio is for people; an HTTP API is for systems. We write a small service to `artifacts/serve.py` (classifier + category + priority, the same logic as `assist()` without the history search, to keep it light) and test it **in-process** with FastAPI's `TestClient`, no server needed. To run it for real: `uvicorn serve:app --app-dir artifacts --port 8000` from `labs/`.

# %%
SERVE = '''"""Ticket classifier service. Run from labs/: uvicorn serve:app --app-dir artifacts --port 8000"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_data import priority_for  # noqa: E402

MODEL_DIR = HERE / "intent-distilmbert"
INFO = json.loads((MODEL_DIR / "intent_labels.json").read_text(encoding="utf-8"))
LABELS = json.loads((HERE.parent / "data" / "labels.json").read_text(encoding="utf-8"))
clf = pipeline("text-classification", model=str(MODEL_DIR), device=-1, truncation=True,
               max_length=INFO["max_length"])  # loaded once per process, not per request

app = FastAPI(title="DemoMart ticket classifier")


class Ticket(BaseModel):
    text: str


class Prediction(BaseModel):
    intent: str
    confidence: float
    category: str
    priority: str
    needs_review: bool


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR.name}


@app.post("/classify", response_model=Prediction)
def classify(ticket: Ticket):
    out = clf(ticket.text)[0]
    return Prediction(intent=out["label"], confidence=round(out["score"], 4),
                      category=LABELS["intent_to_category"][out["label"]],
                      priority=priority_for(out["label"], ticket.text), needs_review=out["score"] < 0.5)
'''
(U.ARTIFACTS / "serve.py").write_text(SERVE, encoding="utf-8")
print("wrote artifacts/serve.py")

# %%
if FT_DIR.exists():
    import importlib.util
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("serve", U.ARTIFACTS / "serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    client = TestClient(serve.app)
    print(client.get("/health").json())
    for t in ["where is my refund", "أريد إلغاء طلبي"]:
        r = client.post("/classify", json={"text": t})
        print(r.status_code, json.dumps(r.json(), ensure_ascii=False))
else:
    print("serve.py needs artifacts/intent-distilmbert (run Lab 3.2 first)")

# %% [markdown]
# ## Checkpoint
#
# 1. Model A has EN macro-F1 0.962 [0.950, 0.973] on 1,080 tickets; model B has 0.975 [0.940, 0.995] on 100 tickets. Is B better?
# 2. A behavioural test fails: "how cn i trak my ordr" gets `delivery_period`. What do you do with that information?
# 3. Why is the reply in `assist()` built from retrieved agent replies rather than generated freely?
#
# <details><summary>Answers</summary>
#
# 1. You can't tell: the intervals overlap and they were measured on different tickets. Compare on the common tickets (or evaluate B on all 1,080).
# 2. It's a coverage gap (typos). Add typo-augmented training examples, re-train, and keep the test in the suite so the bug can't come back.
# 3. Grounding: replies reuse facts that agents actually wrote (Day 2). A free-form LLM reply can invent policies; with retrieval it can still be checked and cited.
#
# </details>
#
# ## Capstone template
#
# Copy this notebook as the evaluation and serving part of your capstone. Before you present, check the reproducibility list from the slides:
#
# - [ ] Pinned package versions (`uv.lock` / `requirements.txt`) and a pinned base-model revision.
# - [ ] Data snapshot, split seed and preprocessing code in the repo (Lab 3.1 shows the pattern).
# - [ ] All hyperparameters in a config file or one config cell, not in your head.
# - [ ] Seeds set; for small data, results over ≥ 3 seeds (mean ± std).
# - [ ] Saved: model + tokenizer + label maps + metrics (Lab 3.2 shows the pattern).
# - [ ] A baseline and at least one improvement **on the same test set**, with CIs (section 1).
# - [ ] ≥ 10 categorized failures (section 2) and a behavioural test suite (section 3).
# - [ ] Latency numbers (p95, batch vs. single) and a working demo (sections 4–5).
# - [ ] A model card in the README: intended use, data, metrics by slice (EN/AR), known limitations, owner.
#
# ## Carry forward
#
# Tomorrow (Day 4) you present your capstone: the leaderboard, error table, behavioural tests and demo from this notebook are the evidence the rubric asks for.
