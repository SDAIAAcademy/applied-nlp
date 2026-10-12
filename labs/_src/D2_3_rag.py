# %% [markdown]
# # Lab 2.3 · Minimal RAG over the help center and past tickets
#
# Day 2 · Session 3 (Retrieval-augmented generation) · 25 min
#
# **What you'll build**
# - A retriever over the DemoMart help center: dense (e5) + BM25, fused with RRF, reranked by a cross-encoder, with Hit@k and MRR numbers for each step.
# - A grounded generator that answers from numbered context, returns `{reply, citations, needs_human}` as validated JSON, and is checked in code.
#
# No framework: about 100 lines you fully understand. LangChain, LlamaIndex and Haystack wrap these same steps.
#
# **Tasks** (matching the lab slide)
# 1. Chunk KB articles by heading; attach metadata (4')
# 2. ✏️ Dense + BM25 retrieval; fuse with RRF (5')
# 3. ✏️ Evaluate Hit@3 on the provided question set (4')
# 4. Rerank with a cross-encoder; re-evaluate (4')
# 5. ✏️ Grounded, cited JSON reply; validate citations (6')
# 6. Ask an unanswerable question: does it abstain? (2')
# 7. ✱ Stretch: Arabic questions against the English KB, with and without query translation
#
# The generator is a small local model (Qwen2.5-1.5B on GPU, 0.5B on CPU). It will sometimes ignore the context, cite the wrong number or forget to abstain; the checks in code are there to catch exactly that. Set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `labs/.env` to send the same calls to a stronger OpenAI-compatible model.

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
import json, re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

pd.set_option("display.max_colwidth", 100)

# %% [markdown]
# ## 1 · Chunk KB articles by heading; attach metadata (4')
#
# The help center is 10 Markdown articles with `## ` sections. Each section answers one question, so **structure-aware chunking** (one section = one chunk) is the natural choice. `U.load_kb()` does the split; we add metadata we'll use to filter and to cite.

# %%
chunks = []
for s in U.load_kb():
    chunks.append({**s, "source": f"kb/{s['doc']}.md", "lang": "en", "n_words": len(s["text"].split()),
                   # what we embed and index: heading path + text (headings carry a lot of meaning)
                   "index_text": f"{s['title']} > {s['section']}\n{s['text']}"})
kb = pd.DataFrame(chunks).set_index("id", drop=False)
print(len(kb), "chunks from", kb.doc.nunique(), "articles | words per chunk:", kb.n_words.describe()[["min", "mean", "max"]].round(0).to_dict())
kb[["title", "section", "source", "n_words"]].head(8)

# %% [markdown]
# For contrast, the naive alternative: fixed windows of 60 words with 15 words of overlap, ignoring headings. Look at where the chunks start and end.

# %%
def fixed_chunks(text: str, size: int = 60, overlap: int = 15) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), size - overlap)]


refunds_md = (U.DATA / "kb" / "refunds.md").read_text(encoding="utf-8")
fixed = fixed_chunks(refunds_md)
print(f"refunds.md: {len(fixed)} fixed-size chunks vs {(kb.doc == 'refunds').sum()} section chunks\n")
for c in fixed[1:3]:
    print("…", c, "…\n")

# %% [markdown]
# 🔍 **Look at**: fixed windows cut sections in half and mix two topics in one chunk ("…within 7 days of receiving the label. ## Refund timelines Refunds are issued…"). Their ids no longer match a section, so citations get vague. Fixed windows are a fine baseline for uniform prose; for help centers, use the structure.

# %% [markdown]
# ## 2 · Dense + BM25 retrieval; fuse with RRF (5')
#
# - **Dense**: e5 embeddings. e5 needs `"passage: "` for documents and `"query: "` for questions.
# - **BM25**: classic keyword scoring. We tokenize with a simple regex (lowercase words); it works for Arabic script too, but an Arabic question shares no words with an English KB.

# %%
chunk_ids = kb.index.tolist()
chunk_vecs = U.embed(kb.index_text, prefix="passage: ")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(t) for t in kb.index_text])


def dense_search(query: str, k: int = 20) -> list[str]:
    scores = chunk_vecs @ U.embed([query], prefix="query: ")[0]
    return [chunk_ids[i] for i in np.argsort(-scores)[:k]]


def bm25_search(query: str, k: int = 20) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    return [chunk_ids[i] for i in np.argsort(-scores, kind="stable")[:k]]


q = "I was charged twice for INV-2024-00731"
print("dense:", dense_search(q)[:3])
print("bm25: ", bm25_search(q)[:3])

# %% [markdown]
# ✏️ **Exercise**: implement **Reciprocal Rank Fusion**: `RRF(d) = Σ 1 / (k + rank_i(d))` with ranks starting at 1, summed over the input rankings. Return the doc ids sorted by fused score, best first. It only uses ranks, so you never need to calibrate BM25 scores against cosine similarities.

# %%
def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    # BEGIN SOLUTION
    scores = {}
    for ranking in rankings:
        for r, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + r)
    return sorted(scores, key=scores.get, reverse=True)
    # END SOLUTION


def hybrid_search(query: str, k: int = 20) -> list[str]:
    return rrf(bm25_search(query, k), dense_search(query, k))[:k]


print("hybrid:", hybrid_search(q)[:3])

# %% [markdown]
# ## 3 · Evaluate Hit@3 on the provided question set (4')
#
# `data/rag_eval.csv` has 32 questions (EN and AR) with the gold section id; 2 are unanswerable (no gold) and are kept for task 6.
#
# - **Hit@k**: is the gold chunk in the top k?
# - **MRR**: mean of 1/rank of the gold chunk (0 if not retrieved). Rewards ranking it *high*.
#
# ✏️ **Exercise**: given `ranks` (the 1-based rank of the gold chunk for each question, or `None` if it was not retrieved), return a dict with `hit@1`, `hit@3` and `mrr`.

# %%
rag_eval = pd.read_csv(U.DATA / "rag_eval.csv").fillna({"gold_section": ""})
answerable = rag_eval[rag_eval.gold_section != ""].reset_index(drop=True)
unanswerable = rag_eval[rag_eval.gold_section == ""].reset_index(drop=True)
print(len(answerable), "answerable questions:", answerable.lang.value_counts().to_dict(), "|", len(unanswerable), "unanswerable")


def rank_of(gold: str, ranking: list[str]):
    return ranking.index(gold) + 1 if gold in ranking else None


def retrieval_metrics(ranks: list) -> dict:
    # BEGIN SOLUTION
    hit = lambda k: np.mean([r is not None and r <= k for r in ranks])
    mrr = np.mean([1 / r if r else 0.0 for r in ranks])
    return {"hit@1": hit(1), "hit@3": hit(3), "mrr": mrr}
    # END SOLUTION

# %%
def evaluate(search_fn, questions: pd.DataFrame = answerable) -> dict:
    ranks = [rank_of(g, search_fn(q)) for q, g in zip(questions.question, questions.gold_section)]
    return retrieval_metrics(ranks)


systems = {"BM25": bm25_search, "dense (e5)": dense_search, "hybrid (RRF)": hybrid_search}
results = {}
for name, fn in systems.items():
    results[name] = {**evaluate(fn),
                     **{f"hit@3 {lang}": evaluate(fn, answerable[answerable.lang == lang])["hit@3"] for lang in ["en", "ar"]}}
pd.DataFrame(results).T.round(3)

# %% [markdown]
# 🔍 **Look at**: BM25 is strong on English questions that reuse KB words, and near zero on Arabic (no shared words). Dense retrieval handles paraphrases and crosses languages. RRF combines them; it can also be slightly *worse* than the best single system when one of the two inputs is noise (BM25 on Arabic). With only 30 questions, one question is 3 points: keep differences in proportion.

# %% [markdown]
# ## 4 · Rerank with a cross-encoder; re-evaluate (4')
#
# A cross-encoder reads the question and a chunk **together** and outputs a relevance score. It is too slow to run over a whole corpus, but perfect for re-sorting the top-20 from the fast retrievers. We use a multilingual model trained on mMARCO (`U.MODELS["reranker"]`).

# %%
reranker = CrossEncoder(U.MODELS["reranker"], device=U.device())


def rerank(query: str, ids: list[str]) -> list[tuple[str, float]]:
    scores = reranker.predict([(query, kb.loc[i, "index_text"]) for i in ids])
    order = np.argsort(-scores)
    return [(ids[i], float(scores[i])) for i in order]


def hybrid_rerank_search(query: str, k: int = 20) -> list[str]:
    return [i for i, _ in rerank(query, hybrid_search(query, k))]


results["hybrid + rerank"] = {**evaluate(hybrid_rerank_search),
                              **{f"hit@3 {lang}": evaluate(hybrid_rerank_search, answerable[answerable.lang == lang])["hit@3"]
                                 for lang in ["en", "ar"]}}
table = pd.DataFrame(results).T.round(3)
display(table)

ax = table[["hit@1", "hit@3", "mrr"]].plot.bar(figsize=(8, 3.5), rot=0, color=["#1c355e", "#00ae8d", "#e96852"])
ax.set_title("Retrieval quality on rag_eval.csv (answerable questions)")
ax.set_xlabel("retriever")
ax.set_ylabel("score")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at**: reranking mostly improves **Hit@1 and MRR** (the right chunk moves to the top), which matters because we will only pass 4 chunks to the LLM. The reranker score is also a useful **abstention signal**: compare the best score for answerable and unanswerable questions.

# %%
top_scores = pd.DataFrame({"question": rag_eval.question,
                           "answerable": rag_eval.gold_section != "",
                           "top_rerank_score": [rerank(q, hybrid_search(q))[0][1] for q in rag_eval.question]})
display(top_scores.groupby("answerable").top_rerank_score.describe()[["min", "50%", "max"]].round(2))
display(top_scores[~top_scores.answerable].round(2))

# %% [markdown]
# ## 5 · Grounded, cited JSON reply; validate citations (6')
#
# The context has numbered items: the top-4 reranked KB chunks and the 2 most similar **resolved tickets** from the history (with the agent's reply). The model must answer only from it, cite `[n]`, and set `needs_human` when the context does not contain the answer.

# %%
llm = U.get_llm()
history = U.load_tickets("history").head(U.pick(8000, 3000))
hist_vecs = U.embed(history.text, cache_key="history")      # cached on disk after the first run


def similar_history(query: str, k: int = 2) -> pd.DataFrame:
    scores = hist_vecs @ U.embed([query])[0]
    return history.iloc[np.argsort(-scores)[:k]]


def build_context(query: str, n_kb: int = 4, n_hist: int = 2) -> tuple[list[dict], float]:
    ranked = rerank(query, hybrid_search(query))
    items = [{"ref": f"{kb.loc[i, 'source']} · {kb.loc[i, 'section']}", "id": i, "text": kb.loc[i, "text"]}
             for i, _ in ranked[:n_kb]]
    for h in similar_history(query, n_hist).itertuples():
        items.append({"ref": f"ticket #{h.ticket_id} · resolved", "id": f"ticket#{h.ticket_id}",
                      "text": f"Customer: {h.text}\nAgent: {h.response[:300]}"})
    return items, ranked[0][1]


def format_context(items: list[dict]) -> str:
    return "\n\n".join(f"[{n}] ({it['ref']})\n{it['text']}" for n, it in enumerate(items, start=1))


items, _ = build_context("How long does a refund to my Visa card take?")
print(format_context(items)[:1200], "…")

# %%
class GroundedReply(BaseModel):
    reply: str = Field(..., max_length=1000)
    citations: list[int] = []          # missing → no citations (caught by the checks below)
    needs_human: bool = True           # missing flag → the safe default: a human looks at it


RAG_SYSTEM = """You draft replies for DemoMart support agents.
Use ONLY the numbered context. Cite sources in the reply as [1], [2] and list the numbers in "citations".
If the context does not contain the answer, set "needs_human": true and say what is missing.
Reply to the customer in the customer's language, in at most 3 sentences.
Respond with one JSON object only: {"reply": "...", "citations": [...], "needs_human": true/false}"""

# Two tiny worked examples (one answer, one abstention): they teach a small model the format and its role.
RAG_EXAMPLES = [
    {"role": "user", "content": "CONTEXT\n[1] (kb/refunds.md · Refund timelines)\nDemoMart Wallet: within 24 hours.\n"
                                "[2] (kb/shipping.md · Delivery options)\nStandard delivery takes 2 to 4 days.\n\n"
                                "CUSTOMER MESSAGE\n<ticket>How fast is a refund to my wallet?</ticket>"},
    {"role": "assistant", "content": '{"reply": "Refunds to your DemoMart Wallet arrive within 24 hours [1].", '
                                     '"citations": [1], "needs_human": false}'},
    {"role": "user", "content": "CONTEXT\n[1] (kb/shipping.md · Delivery options)\nStandard delivery takes 2 to 4 days.\n\n"
                                "CUSTOMER MESSAGE\n<ticket>Can I pay with Bitcoin?</ticket>"},
    {"role": "assistant", "content": '{"reply": "Thank you for asking. I could not find information about Bitcoin '
                                     'payments, so a colleague will follow up.", "citations": [], "needs_human": true}'},
]


def rag_messages(query: str, items: list[dict]) -> list[dict]:
    return [{"role": "system", "content": RAG_SYSTEM}] + RAG_EXAMPLES + [
        {"role": "user", "content": f"CONTEXT\n{format_context(items)}\n\nCUSTOMER MESSAGE\n<ticket>{query}</ticket>"}]

# %% [markdown]
# Schema validation checks the *shape*. Citations need a check against *this* context.
#
# ✏️ **Exercise**: write `citation_problems(ans, n_items)` returning a list of problems (empty = OK):
# - a cited number that is not between 1 and `n_items` → `"unknown citation k"`;
# - no citations while `needs_human` is false → `"no citations"`;
# - a `[k]` written in the reply text but missing from `citations` → `"[k] not listed"`.

# %%
def citation_problems(ans: GroundedReply, n_items: int) -> list[str]:
    problems = []
    # BEGIN SOLUTION
    problems += [f"unknown citation {c}" for c in ans.citations if not 1 <= c <= n_items]
    if not ans.citations and not ans.needs_human:
        problems.append("no citations")
    in_text = {int(k) for k in re.findall(r"\[(\d+)\]", ans.reply)}
    problems += [f"[{k}] not listed" for k in sorted(in_text - set(ans.citations))]
    # END SOLUTION
    return problems

# %% [markdown]
# The full answer step: retrieve, generate, validate (one retry with the error message, as in Lab 2.2), check citations. `first_json` keeps the first complete JSON object, because small models sometimes keep writing after it. We also add a guard in code: if even the best reranked chunk scores low, we set `needs_human` regardless of what the model says. `MIN_RERANK` comes from the score table above: unanswerable questions score around -5 or lower. A few answerable questions also score low, so this guard sends them to a human too; that's the price of a safe default. Treat it as a starting point, and tune it on more questions than we have here.

# %%
MIN_RERANK = -4.0


def first_json(text: str) -> str:
    """The first complete JSON object in `text`: tolerates fences, chatter and trailing junk."""
    start = text.find("{")
    if start < 0:
        return text
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        return text[start:]


def answer(query: str) -> dict:
    items, top_score = build_context(query)
    messages = rag_messages(query, items)
    ans, raw = None, ""
    for _ in range(2):                                   # first try + one retry
        raw = llm.chat(messages, temperature=0, max_new_tokens=U.pick(200, 150))
        try:
            ans = GroundedReply.model_validate_json(first_json(raw))
            break
        except ValidationError as e:
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": f"Invalid output: {str(e)[:300]}. Return corrected JSON only."}]
    out = {"question": query, "context_ids": [it["id"] for it in items], "top_rerank": round(top_score, 2)}
    if ans is None:
        return {**out, "valid": False, "reply": raw[:200], "citations": [], "model_needs_human": None,
                "needs_human": True, "problems": ["invalid JSON"]}
    problems = citation_problems(ans, len(items))
    needs_human = ans.needs_human or top_score < MIN_RERANK or bool(problems)
    model_flag = ans.needs_human if "needs_human" in ans.model_fields_set else None   # None: the model omitted it
    return {**out, "valid": True, "reply": ans.reply, "citations": ans.citations, "model_needs_human": model_flag,
            "needs_human": needs_human, "problems": problems}

# %%
demo = pd.concat([answerable[answerable.lang == "en"].iloc[[0, 13, 7, 20]].head(U.pick(4, 2)),
                  answerable[answerable.lang == "ar"].head(U.pick(2, 1))])
rows = []
for r in demo.itertuples():
    out = answer(r.question)
    out["gold_in_context"] = r.gold_section in out["context_ids"]
    rows.append(out)
answers = pd.DataFrame(rows)
print(f"valid JSON: {answers.valid.mean():.0%} | citation problems: {(answers.problems.str.len() > 0).mean():.0%} "
      f"| gold chunk in context: {answers.gold_in_context.mean():.0%}")
answers[["question", "reply", "citations", "needs_human", "problems", "gold_in_context"]]

# %% [markdown]
# 🔍 **Look at**: read each reply next to its context. Does the refund answer say "5 to 7 business days" for cards? Is the citation the chunk that actually says it? A valid, cited reply can still be unfaithful (citing [1] for a claim that's in [3], or not in the context at all). We'll grade that in Lab 2.4. Does the Arabic reply come out in Arabic?

# %% [markdown]
# ## 6 · Ask an unanswerable question: does it abstain? (2')
#
# The KB has nothing about gaming laptops or the stock price. A good assistant says so and hands over to a human, instead of inventing an answer.

# %%
abstain = pd.DataFrame([answer(q) for q in unanswerable.question])
abstain[["question", "top_rerank", "reply", "model_needs_human", "needs_human", "problems"]]

# %% [markdown]
# 🔍 **Look at**: `model_needs_human` is what the LLM decided (`None` if it left the field out, in which case the schema's safe default `True` applies); `needs_human` adds the code checks (low reranker score, citation problems). Small models often answer anyway, with a polite but unsupported reply. Don't rely on the model alone to abstain: combine its flag with retrieval scores and citation checks.

# %% [markdown]
# ## ✱ Stretch · Arabic questions against the English KB, with and without translation
#
# Query rewriting: translate the Arabic question to English with the LLM, then search. Compare Hit@3 on the Arabic questions. Dense retrieval is cross-lingual already, so the gain shows up mostly for BM25 and hybrid.

# %%
ar_q = answerable[answerable.lang == "ar"].reset_index(drop=True)


def translate(text: str) -> str:
    msgs = [{"role": "system", "content": "Translate the customer message to English. Output only the translation."},
            {"role": "user", "content": text}]
    return llm.chat(msgs, temperature=0, max_new_tokens=40).split("\n")[0]


ar_q["translated"] = [translate(q) for q in ar_q.question]
display(ar_q[["question", "translated", "gold_section"]])

rows = {}
for name, fn in {"BM25": bm25_search, "dense (e5)": dense_search, "hybrid + rerank": hybrid_rerank_search}.items():
    rows[name] = {"hit@3 Arabic as is": evaluate(fn, ar_q)["hit@3"],
                  "hit@3 translated": evaluate(fn, ar_q.assign(question=ar_q.translated))["hit@3"]}
pd.DataFrame(rows).T.round(3)

# %% [markdown]
# 🔍 **Look at**: translation costs one extra (short) LLM call per ticket. Is it worth it here? With a multilingual embedder and reranker the gap is small; with an English-only retriever or BM25 it is large. Check the translations: a small model sometimes paraphrases away the key term.

# %% [markdown]
# ## Checkpoint
#
# 1. Hit@3 is 0.95 but replies are often wrong. Which stage do you investigate, and how?
# 2. Why rerank only the top-20 instead of the whole KB?
# 3. Name two checks in code that catch a reply the model should not have sent.
#
# <details><summary>Answers</summary>
#
# 1. Generation: retrieval already finds the evidence. Check whether the gold chunk is in the context but the reply contradicts or ignores it; tighten grounding rules, reduce or reorder context, or use a stronger model.
# 2. A cross-encoder runs a full transformer pass per (question, chunk) pair; it's too slow for a large corpus. Fast retrievers narrow it to 20 candidates; the reranker sorts those precisely.
# 3. Schema validation; citations must exist in the context (and the reply must cite something); a minimum reranker score; language check; needs_human routing.
#
# </details>
#
# ## Carry forward
#
# Lab 2.4 plugs `hybrid search → rerank → grounded reply → citation checks` into `assist(ticket)`, together with the classifier and the extraction from Lab 2.2, and grades the replies for faithfulness.
