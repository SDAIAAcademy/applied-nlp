# %% [markdown]
# # Lab 1.2 · Build semantic search over historical tickets
#
# Day 1 · Session 2 (Embeddings & semantic search) · 25 min. What you'll build:
# - sentence embeddings for English and Arabic tickets with `intfloat/multilingual-e5-small`, and the pooling step done by hand;
# - a semantic search over 8,000 resolved tickets (NumPy, then FAISS);
# - a number on retrieval quality (Hit@5, MRR) for English queries and for Arabic queries against the English history.
#
# **Tasks**
# - [ ] 1 · Embed sentences; compute a similarity matrix and a heatmap (5')
# - [ ] 2 · Compare mean pooling by hand with the library output (5')
# - [ ] 3 · Index the historical tickets with NumPy, then FAISS (5')
# - [ ] 4 · Evaluate Hit@5 and MRR on test tickets (same-intent relevance) (5')
# - [ ] 5 · Cross-lingual: Arabic queries → English history (5')
# - [ ] ✱ Stretch · Compare two embedding models, or plot a 2-D UMAP of intents

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

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

pd.set_option("display.max_colwidth", 90)

history = U.load_tickets("history")              # sorted by intent: subsample randomly, never with .head()
if U.QUICK:
    history = history.sample(1000, random_state=0).sort_index().reset_index(drop=True)
test = U.load_tickets("test")
ar = U.load_tickets("ar_test")
print(len(history), "history tickets |", len(test), "EN test |", len(ar), "AR test")

# %% [markdown]
# ## 1 · Embed sentences; compute a similarity matrix and a heatmap (5')
#
# `multilingual-e5-small` maps text in ~100 languages into one 384-dimensional space. It was trained with
# prefixes: `"query: "` for questions and short texts, `"passage: "` for documents to be retrieved. Forgetting them
# silently lowers quality.
#
# `U.get_embedder()` returns a cached `SentenceTransformer("intfloat/multilingual-e5-small")`.
# With `normalize_embeddings=True` every vector has length 1, so cosine similarity is a plain dot product.

# %%
model = U.get_embedder()  # SentenceTransformer(U.MODELS["embed"])

sentences = [
    ("EN refund",   "I want a refund for my order"),
    ("AR refund",   "لم يصلني المبلغ المسترد حتى الآن"),
    ("EN delivery", "Where is my package? It has not arrived"),
    ("AR delivery", "وين طلبي؟ ما وصل لين الحين"),
    ("EN password", "I forgot my password and cannot log in"),
    ("AR password", "نسيت كلمة المرور ولا أستطيع الدخول"),
    ("EN payment",  "You charged me twice for one order"),
    ("AR payment",  "تم خصم المبلغ مرتين من بطاقتي"),
]
names = [n for n, _ in sentences]
texts = [t for _, t in sentences]

emb = model.encode(["query: " + t for t in texts], normalize_embeddings=True)
print("shape:", emb.shape, "| norms:", np.linalg.norm(emb, axis=1).round(3))
sim = emb @ emb.T

# %%
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(sim, cmap="viridis")
ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if sim[i, j] < sim.max() * 0.9 else "black")
ax.set_xlabel("sentence")
ax.set_ylabel("sentence")
ax.set_title("Cosine similarity, multilingual-e5-small")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at** three things. (1) Translation pairs: the password and payment pairs are the best match in their rows,
# so English and Arabic share one space. (2) Where it is weaker: "EN refund" is closer to "EN payment" (both about money)
# than to its Arabic version, and the Gulf-dialect delivery sentence matches nothing well. (3) The absolute numbers:
# e5 scores almost every pair between 0.75 and 1.0.
# Scores are relative; compare rankings, never a raw value from one model with another.
#
# ## 2 · Compare mean pooling by hand with the library output (5')
#
# A sentence embedding is the transformer's token vectors pooled into one vector. For e5 the pooling is the
# **mean over real tokens** (padding excluded via the attention mask), followed by L2 normalization.
# Let's reproduce it with plain `transformers`.

# %%
hf_tok = AutoTokenizer.from_pretrained(U.MODELS["embed"])
hf_model = AutoModel.from_pretrained(U.MODELS["embed"]).eval()

enc = hf_tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    hidden = hf_model(**enc).last_hidden_state      # (n_sentences, seq_len, 384)
mask = enc["attention_mask"]                         # (n_sentences, seq_len), 1 = real token
print("hidden:", tuple(hidden.shape), "| mask:", tuple(mask.shape), "| real tokens per sentence:", mask.sum(1).tolist())

# %% [markdown]
# ✏️ **Exercise** · Complete `mean_pool(hidden, mask)`: zero out padded positions, sum over the sequence axis and divide
# by the number of real tokens. Shapes: `hidden` is `(n, seq, d)`, `mask` is `(n, seq)`, the result is `(n, d)`.

# %%
def mean_pool(hidden, mask):
    # BEGIN SOLUTION
    m = mask.unsqueeze(-1).float()                   # (n, seq, 1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
    # END SOLUTION

# %%
# Check + safety net
try:
    pooled = mean_pool(hidden, mask)
except NotImplementedError:
    print("Exercise skipped: using the reference mean_pool()")
    def mean_pool(hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1)
    pooled = mean_pool(hidden, mask)

manual = torch.nn.functional.normalize(pooled, dim=-1).numpy()
naive = torch.nn.functional.normalize(hidden.mean(dim=1), dim=-1).numpy()   # mean over padding too
pd.DataFrame({
    "real tokens": mask.sum(1).numpy(),
    "cos(manual, library)": (manual * emb).sum(1),
    "cos(naive mean incl. padding, library)": (naive * emb).sum(1),
}, index=names).round(4)

# %% [markdown]
# The masked mean reproduces the library output (cosine ≈ 1.0). The naive mean that includes padding drifts for
# the shortest sentences, which get the most padding. This bug appears whenever someone pools by hand and forgets the mask.
#
# ## 3 · Index the historical tickets with NumPy, then FAISS (5')
#
# Offline: embed every resolved ticket once with the `"passage: "` prefix and keep the vectors.
# Online: embed the new ticket with `"query: "` and rank the corpus by dot product.
# `U.embed(..., cache_key=...)` stores the vectors in `labs/cache/`, so the next labs reuse them.

# %%
t0 = time.perf_counter()
doc_emb = U.embed(history.text, prefix="passage: ", cache_key="history")
print(f"doc_emb: {doc_emb.shape} {doc_emb.dtype} in {time.perf_counter() - t0:.1f}s (cached after the first run)")

# %% [markdown]
# ✏️ **Exercise** · Complete `top_k(q, doc_emb, k)`: score every document with one matrix-vector product and return the
# indices and scores of the `k` best, best first (`np.argsort` on the negated scores).

# %%
def top_k(q, doc_emb, k=5):
    """q: (d,) normalized query vector. Returns (indices, scores), best first."""
    # BEGIN SOLUTION
    scores = doc_emb @ q
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]
    # END SOLUTION

# %%
# Check + safety net
try:
    _idx, _sc = top_k(doc_emb[3], doc_emb, k=3)
    assert _idx[0] == 3 and np.all(np.diff(_sc) <= 0), "the document itself should be its own best match"
    print("top_k() looks right")
except NotImplementedError:
    print("Exercise skipped: using the reference top_k()")
    def top_k(q, doc_emb, k=5):
        scores = doc_emb @ q
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]

def search_np(query, k=5):
    q = U.embed([query], prefix="query: ")[0]
    idx, scores = top_k(q, doc_emb, k)
    return history.iloc[idx][["text", "intent"]].assign(score=scores.round(3))

search_np("I was charged two times for the same purchase")

# %% [markdown]
# Exact (brute-force) search over 8,000 vectors is a single matrix product: milliseconds. FAISS gives the same
# result with an index object you can save, extend, and later swap for an approximate index (IVF, HNSW) when
# you reach millions of vectors. `IndexFlatIP` = exact inner product = cosine on normalized vectors.

# %%
index = faiss.IndexFlatIP(doc_emb.shape[1])
index.add(doc_emb)
print("vectors in index:", index.ntotal)

def search(query, k=5):
    q = U.embed([query], prefix="query: ")
    scores, ids = index.search(q, k)
    return history.iloc[ids[0]][["text", "intent"]].assign(score=scores[0].round(3))

search("لم يصلني المبلغ المسترد")   # Arabic query, English history

# %%
# Same ranking from NumPy and FAISS? And how fast is a batch of 200 queries?
q_batch = U.embed(test.text[:200], prefix="query: ")
t0 = time.perf_counter(); np_ids = np.argsort(-(q_batch @ doc_emb.T), axis=1)[:, :5]; t_np = time.perf_counter() - t0
t0 = time.perf_counter(); _, faiss_ids = index.search(q_batch, 5); t_faiss = time.perf_counter() - t0
print(f"identical top-5: {(np_ids == faiss_ids).mean():.1%} | NumPy {t_np * 1000:.1f} ms | FAISS {t_faiss * 1000:.1f} ms for 200 queries")

# %% [markdown]
# (Ties between near-duplicate tickets can swap order, so a value slightly below 100% is fine.)
#
# ## 4 · Evaluate Hit@5 and MRR on test tickets (same-intent relevance) (5')
#
# Cheap relevance labels: a retrieved history ticket counts as relevant if it has the **same intent** as the query.
# Test tickets are not in the history index, so there is no leakage.
#
# - **Hit@k**: share of queries with at least one relevant result in the top k.
# - **MRR** (mean reciprocal rank): average of 1/rank of the first relevant result (0 if none in the top k).
#
# We retrieve the top 10 for every query and build a boolean matrix `rel` of shape `(n_queries, 10)`.

# %%
test_emb = U.embed(test.text, prefix="query: ", cache_key="test")     # all EN test tickets (cached, reused in Lab 1.4)
rng = np.random.default_rng(0)
q_rows = rng.choice(len(test), size=U.pick(500, 50), replace=False)
queries = test.iloc[q_rows].reset_index(drop=True)
q_emb = test_emb[q_rows]

def retrieve(q_emb, k=10, index=index):
    _, ids = index.search(q_emb, k)
    return ids

ids = retrieve(q_emb)
rel = history.intent.to_numpy()[ids] == queries.intent.to_numpy()[:, None]
print("rel:", rel.shape, "| first query:", queries.text[0], "->", rel[0].astype(int))

# %% [markdown]
# ✏️ **Exercise** · Complete `mrr(rel)`. Tip: `rel.argmax(axis=1)` gives the position of the first `True` in each row
# (and 0 when the row has no `True`, so check `rel.any(axis=1)`).

# %%
def hit_at_k(rel, k=5):
    return rel[:, :k].any(axis=1).mean()

def mrr(rel):
    # BEGIN SOLUTION
    first = rel.argmax(axis=1)
    rr = np.where(rel.any(axis=1), 1.0 / (first + 1), 0.0)
    return rr.mean()
    # END SOLUTION

# %%
# Check + safety net
try:
    toy = np.array([[True, False, False], [False, False, True], [False, False, False]])
    assert abs(mrr(toy) - (1 + 1 / 3 + 0) / 3) < 1e-9
    print("mrr() looks right")
except NotImplementedError:
    print("Exercise skipped: using the reference mrr()")
    def mrr(rel):
        return np.where(rel.any(axis=1), 1.0 / (rel.argmax(axis=1) + 1), 0.0).mean()

def evaluate(q_emb, q_intents, index=index, labels=None):
    labels = history.intent.to_numpy() if labels is None else labels
    rel = labels[retrieve(q_emb, 10, index)] == np.asarray(q_intents)[:, None]
    return {"n": len(rel), "Hit@1": hit_at_k(rel, 1), "Hit@5": hit_at_k(rel, 5),
            "P@5": rel[:, :5].mean(), "MRR@10": mrr(rel)}

pd.DataFrame([evaluate(q_emb, queries.intent)], index=["EN test → EN history"]).round(3)

# %% [markdown]
# 🔍 **Look at** the gap between Hit@5 and Hit@1. Bitext tickets are templated, so English scores are high;
# real tickets are messier. Which queries fail? Usually near-identical intents (`get_refund` vs `track_refund`).

# %%
misses = ~rel[:, :1].ravel()
pd.DataFrame({
    "query": queries.text[misses],
    "intent": queries.intent[misses],
    "top-1 intent": history.intent.to_numpy()[ids[misses, 0]],
}).head(8)

# %% [markdown]
# ## 5 · Cross-lingual: Arabic queries → English history (5')
#
# Same index, same model; now the queries are the 108 Arabic tickets (MSA and Gulf dialect).
# A multilingual model puts both languages in one space, so this works without translation. But it has a cost.

# %%
ar_emb = U.embed(ar.text, prefix="query: ", cache_key="ar_test")
results = {"EN test (sample)": evaluate(q_emb, queries.intent), "AR all": evaluate(ar_emb, ar.intent)}
for dialect, part in ar.groupby("dialect"):
    results[f"AR {dialect}"] = evaluate(ar_emb[part.index.to_numpy()], part.intent)
pd.DataFrame(results).T.round(3)

# %%
for i in [0, 9, 60]:
    print(f"\nquery [{ar.dialect[i]}, {ar.intent[i]}]: {ar.text[i]}")
    _, top = index.search(ar_emb[i:i + 1], 3)
    for j in top[0]:
        print(f"   {history.intent[j]:25s} {history.text[j][:80]}")

# %% [markdown]
# 🔍 **Look at** the drop from English to Arabic, and from MSA to Gulf dialect. Part of it is the language gap,
# part is that our Arabic tickets are hand-written and less templated than Bitext. Either way: cross-lingual
# retrieval works "for free", but measure it per language and dialect before you rely on it.
#
# ## ✱ Stretch · Compare two embedding models, or plot a 2-D UMAP of intents
#
# **Option A.** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` is a symmetric paraphrase model:
# no prefixes. Never mix vectors from two models in one index: we build a second index.

# %%
ALT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
doc_alt = U.embed(history.text, prefix="", model=ALT, cache_key="history")
index_alt = faiss.IndexFlatIP(doc_alt.shape[1])
index_alt.add(doc_alt)
q_alt = U.embed(queries.text, prefix="", model=ALT)
ar_alt = U.embed(ar.text, prefix="", model=ALT, cache_key="ar_test")

rows = {}
for model_name, idx_, qe, ae in [("e5-small", index, q_emb, ar_emb), ("paraphrase-mMiniLM", index_alt, q_alt, ar_alt)]:
    rows[(model_name, "EN")] = evaluate(qe, queries.intent, index=idx_)
    rows[(model_name, "AR")] = evaluate(ae, ar.intent, index=idx_)
pd.DataFrame(rows).T.round(3)

# %% [markdown]
# 🔍 Our default is not automatically the best on *this* task: the paraphrase model is trained for exactly this kind of
# short, symmetric "same meaning?" comparison and does at least as well here, especially on Arabic. Leaderboards (MTEB) give
# you candidates; your own evaluation decides. Remember the sample is small (50 EN queries in quick mode), so small
# differences are noise.
#
# **Option B.** A 2-D UMAP projection of the training tickets, coloured by category. Projections distort distances,
# so use them for intuition only.

# %%
import umap

train = U.load_tickets("train")
train_emb = U.embed(train.text, prefix="query: ", cache_key="train")   # cached, reused in Lab 1.4
sub = np.random.default_rng(0).choice(len(train), size=U.pick(len(train), 1500), replace=False)
xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(train_emb[sub])

cats = train.category.to_numpy()[sub]
fig, ax = plt.subplots(figsize=(8, 6))
cmap = plt.get_cmap("tab20")
for i, c in enumerate(sorted(set(cats))):
    m = cats == c
    ax.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(i), label=c)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("Training tickets (e5-small, UMAP), coloured by category")
ax.legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Checkpoint
#
# 1. Why does `doc_emb @ q` give cosine similarity, and what would break if you forgot to normalize?
# 2. You evaluated with "same intent = relevant". Name one way this label overestimates and one way it underestimates real search quality.
# 3. A customer searches for "invoice INV-2024-00731". What does semantic search return, and what would you add?
#
# <details><summary>Answers</summary>
#
# 1. For unit vectors, a·b = cos(a, b). Without normalization, long vectors (not more relevant ones) get higher scores.
# 2. Overestimates: any ticket with the same intent counts as a hit even if it does not help with this specific case.
#    Underestimates: a ticket with a neighbouring intent (e.g. `track_refund` for a `get_refund` query) may be useful but counts as a miss.
# 3. Other invoice tickets, probably not the one with that ID: embeddings capture "about invoices", not exact identifiers.
#    Add keyword search (BM25) and merge the two result lists (hybrid search).
#
# </details>
#
# ## Carry forward
#
# - The cached history, train, test and Arabic embeddings (`labs/cache/`) are reused in Lab 1.4 for the classifier and
#   for `process_ticket()`, which returns the top-3 similar tickets with their responses.
# - Day 2 RAG uses the same embed → index → top-k pattern over the help-center articles, plus a reranker.
