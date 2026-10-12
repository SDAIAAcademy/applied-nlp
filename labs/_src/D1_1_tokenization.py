# %% [markdown]
# # Lab 1.1 · Inspect English and Arabic tokenization
#
# Day 1 · Session 1 (Tokenization) · 25 min. What you'll build:
# - a hands-on feel for what a tokenizer turns a support ticket into (IDs, special tokens, masks);
# - a fertility table that puts a number on the "Arabic token tax" for five tokenizers;
# - an Arabic normalizer and a data-driven `max_length` for the classifiers of Day 1 and Day 3.
#
# **Tasks**
# - [ ] 1 · Tokenize a ticket, map IDs ↔ tokens, decode back (5')
# - [ ] 2 · Inspect special tokens, padding, attention masks (5')
# - [ ] 3 · Compute fertility for EN vs. AR across 5 tokenizers (8')
# - [ ] 4 · Write an Arabic normalizer; measure its effect (5')
# - [ ] 5 · Find the token-length percentile for `max_length` (2')
# - [ ] ✱ Stretch · Train a tiny BPE tokenizer on the tickets

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
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

pd.set_option("display.max_colwidth", 120)

train = U.load_tickets("train")
test = U.load_tickets("test")
ar = U.load_tickets("ar_test")
print(len(train), "train |", len(test), "test |", len(ar), "Arabic test tickets")

# %% [markdown]
# ## 1 · Tokenize a ticket, map IDs ↔ tokens, decode back (5')
#
# A tokenizer is a fixed lookup from text pieces to integers. The model only ever sees the integers.
# We start with XLM-R's tokenizer (SentencePiece Unigram, 250k vocabulary), which we will use again in the next tasks.
#
# > Model names: we use the canonical Hub ids `FacebookAI/xlm-roberta-base` and `google-bert/bert-base-multilingual-cased`.
# > The short legacy names (`xlm-roberta-base`, `bert-base-multilingual-cased`) point to the same repos.

# %%
XLMR = "FacebookAI/xlm-roberta-base"
tok = AutoTokenizer.from_pretrained(XLMR)

ticket = "I have a problem with order A-12345, you charged me twice!"
ids = tok(ticket)["input_ids"]
tokens = tok.convert_ids_to_tokens(ids)

print("text  :", ticket)
print("n_ids :", len(ids), "| n_words:", len(ticket.split()))
pd.DataFrame({"token": tokens, "id": ids}).T

# %% [markdown]
# Going back: `decode` joins the pieces and removes the `▁` word-start markers. With `skip_special_tokens=True`
# you get the original text back (SentencePiece is lossless for normal text).

# %%
print(tok.decode(ids))
print(tok.decode(ids, skip_special_tokens=True))
print("round trip ok:", tok.decode(ids, skip_special_tokens=True) == ticket)
print("id of '▁refund':", tok.convert_tokens_to_ids("▁refund"), "| id 12989 is:", tok.convert_ids_to_tokens(12989))

# %% [markdown]
# Every model family has its own conventions (slide "Same text, different tokenizers"):
# WordPiece marks continuation pieces with `##`, byte-level BPE marks a leading space with `Ġ`,
# SentencePiece marks word starts with `▁`.

# %%
for name in ["google-bert/bert-base-multilingual-cased", "gpt2", XLMR]:
    t = AutoTokenizer.from_pretrained(name)
    print(f"{name:42s}", t.convert_ids_to_tokens(t("unsubscribing from the newsletter")["input_ids"]))

# %% [markdown]
# 🔍 **Look at** how the order ID `A-12345` is split. Identifiers are cut into arbitrary digit chunks,
# which is one reason embeddings blur exact IDs (Lab 1.2).
#
# ## 2 · Inspect special tokens, padding, attention masks (5')
#
# Models are fed batches, so sequences of different length are padded to a common length.
# The `attention_mask` tells the model which positions are real (1) and which are padding (0).
# This is the code from the slide "IDs, masks, padding and truncation".

# %%
batch = tok(
    ["Where is my order?", "أين طلبي؟ لم يصل بعد"],
    padding=True,          # pad to longest in batch
    truncation=True,       # cut at max_length
    max_length=32,
    return_tensors="pt",
)
print(batch["input_ids"].shape)
print(batch["attention_mask"])
for row in batch["input_ids"]:
    print(tok.convert_ids_to_tokens(row))
print("model_max_length:", tok.model_max_length)
print("special tokens  :", dict(zip(tok.all_special_tokens, tok.all_special_ids)))

# %% [markdown]
# Truncation is silent data loss. Cut the same pair at `max_length=6` and see what disappears:
# the special tokens `<s>` / `</s>` are kept, the end of the message is not.

# %%
short = tok(["Where is my order?", "أين طلبي؟ لم يصل بعد"], padding=True, truncation=True, max_length=6)
for row in short["input_ids"]:
    print(tok.convert_ids_to_tokens(row), "->", tok.decode(row, skip_special_tokens=True))

# %% [markdown]
# ## 3 · Compute fertility for EN vs. AR across 5 tokenizers (8')
#
# **Fertility** = number of tokens / number of whitespace words. Lower means cheaper API calls, faster generation
# and more text in the context window. We count tokens without special tokens (`add_special_tokens=False`),
# so that the number reflects the text only.
#
# ✏️ **Exercise** · Complete `fertility(tok, texts)`: total tokens over all texts divided by total whitespace words.

# %%
def fertility(tok, texts):
    """Tokens per whitespace word over a list of texts (no special tokens)."""
    # BEGIN SOLUTION
    n_tokens = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    return n_tokens / n_words
    # END SOLUTION

# %%
# Check (and safety net: if you skipped the exercise, a reference version is used so the lab keeps running)
try:
    two = ["Where is my order?", "أين طلبي؟ لم يصل بعد"]
    expected = sum(len(tok.tokenize(t)) for t in two) / 9  # 9 whitespace words
    assert abs(fertility(tok, two) - expected) < 1e-9, (fertility(tok, two), expected)
    print("fertility() looks right:", round(expected, 3))
except NotImplementedError:
    print("Exercise skipped: using the reference fertility()")
    def fertility(tok, texts):
        return sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts) / sum(len(t.split()) for t in texts)

# %% [markdown]
# Now the table from the slide "Fertility: tokens per word". English: a random sample of test tickets.
# Arabic: all 108 hand-written MSA and Gulf tickets.

# %%
TOKENIZERS = [
    "gpt2",
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
    "aubmindlab/bert-base-arabertv02",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
toks = {name: AutoTokenizer.from_pretrained(name) for name in TOKENIZERS}

en_texts = test.text.sample(U.pick(500, 200), random_state=0).tolist()
ar_texts = ar.text.tolist()

rows = []
for name, t in toks.items():
    rows.append({"tokenizer": name, "vocab": len(t),
                 "EN fertility": fertility(t, en_texts), "AR fertility": fertility(t, ar_texts)})
fert = pd.DataFrame(rows).set_index("tokenizer")
fert["Arabic token tax (AR/EN)"] = fert["AR fertility"] / fert["EN fertility"]
fert.round(2)

# %%
x = np.arange(len(fert))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 0.2, fert["EN fertility"], width=0.4, label="English")
ax.bar(x + 0.2, fert["AR fertility"], width=0.4, label="Arabic")
ax.set_xticks(x, [n.split("/")[-1] for n in fert.index], rotation=15)
ax.axhline(1, color="grey", lw=0.8, ls="--")
ax.set_ylabel("tokens per whitespace word")
ax.set_xlabel("tokenizer")
ax.set_title("Fertility on support tickets: English vs. Arabic")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at** GPT-2: its byte-level BPE was trained mostly on English, so an Arabic word (2 bytes per letter in UTF-8)
# falls apart into many byte pieces. Same message, about 5× the bill and 5× less text in the context window.
# Multilingual vocabularies (XLM-R, Qwen) shrink the gap, and AraBERT is the mirror image: excellent on Arabic, poor on English.
#
# Seeing it on one sentence makes it concrete:

# %%
sample_ar = "لم يصلني المبلغ المسترد حتى الآن"
for name in ["gpt2", "FacebookAI/xlm-roberta-base", "aubmindlab/bert-base-arabertv02"]:
    pieces = toks[name].tokenize(sample_ar)
    print(f"{name:34s} {len(pieces):3d} tokens  {pieces[:12]}")

# %% [markdown]
# ## 4 · Write an Arabic normalizer; measure its effect (5')
#
# The same Arabic word can be written several ways (slide "Arabic is different"). For search and matching we
# usually fold these variants together:
#
# | Variation | Rule |
# |---|---|
# | Alef forms أ إ آ | → ا |
# | Alef maqsura ى | → ي |
# | Diacritics (tashkeel, U+064B–U+0652, plus dagger alef U+0670) | remove |
# | Tatweel ـ (U+0640) | remove |
# | Taa marbuta ة | → ه, **only if** `taa_marbuta=True` (it changes meaning more often) |
#
# ✏️ **Exercise** · Implement `normalize_ar(text, taa_marbuta=False)` with `re.sub` / `str.replace`.

# %%
DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")  # harakat, tanween, shadda, sukun, dagger alef

def normalize_ar(text, taa_marbuta=False):
    # BEGIN SOLUTION
    text = DIACRITICS.sub("", text)
    text = text.replace("\u0640", "")         # tatweel
    text = re.sub("[إأآ]", "ا", text)          # alef forms
    text = text.replace("ى", "ي")              # alef maqsura
    if taa_marbuta:
        text = text.replace("ة", "ه")
    return text
    # END SOLUTION

# %%
# Check + safety net
try:
    assert normalize_ar("طَلَبٌ") == "طلب"
    assert normalize_ar("طـــلب") == "طلب"
    assert normalize_ar("أين إلغاء آخر") == "اين الغاء اخر"
    assert normalize_ar("مستشفى") == "مستشفي"
    assert normalize_ar("فاتورة") == "فاتورة" and normalize_ar("فاتورة", taa_marbuta=True) == "فاتوره"
    print("normalize_ar() passes all checks")
except NotImplementedError:
    print("Exercise skipped: using the reference normalize_ar()")
    def normalize_ar(text, taa_marbuta=False):
        text = re.sub("[إأآ]", "ا", DIACRITICS.sub("", text).replace("ـ", "")).replace("ى", "ي")
        return text.replace("ة", "ه") if taa_marbuta else text

# %% [markdown]
# Variants that a user might type for "where is my order?" collapse to one string after normalization:

# %%
variants = ["أين طلبي؟", "اين طلبي؟", "أَيْنَ طَلَبِي؟", "أيـــن طلبي؟"]
for v in variants:
    print(f"{v:18s} -> {normalize_ar(v):12s} xlm-r: {toks[XLMR].tokenize(v)}")
print("distinct before:", len(set(variants)), "| after:", len({normalize_ar(v) for v in variants}))

# %% [markdown]
# 🔍 **Look at** the fully vowelled version: XLM-R splits it into 14 pieces instead of 4, because diacritics are rare in
# its training data. Removing them is the normalization step with the biggest effect on token counts.

# %% [markdown]
# Now measure the effect on the whole Arabic test set: how many distinct words remain, and what happens to
# fertility and to the number of distinct tokens with AraBERT and XLM-R.

# %%
ar_norm = [normalize_ar(t) for t in ar_texts]
ar_norm_tm = [normalize_ar(t, taa_marbuta=True) for t in ar_texts]

def n_unique_tokens(t, texts):
    return len({p for x in texts for p in t.tokenize(x)})

rows = []
for label, texts in [("raw", ar_texts), ("normalized", ar_norm), ("normalized + taa marbuta", ar_norm_tm)]:
    row = {"version": label, "unique words": len({w for x in texts for w in x.split()})}
    for name in ["aubmindlab/bert-base-arabertv02", XLMR]:
        short_name = name.split("/")[-1]
        row[f"fertility {short_name}"] = fertility(toks[name], texts)
        row[f"unique tokens {short_name}"] = n_unique_tokens(toks[name], texts)
    rows.append(row)
pd.DataFrame(rows).set_index("version").round(3)

# %% [markdown]
# 🔍 **Look at** the direction of the change. Normalization merges spelling variants (fewer unique words and tokens),
# but fertility barely moves and can even go *up* slightly: the tokenizers were trained on raw text that contains
# hamza forms like أ, so a "normalized" spelling can be a rarer string for them.
# That is the slide's warning in numbers: normalize for search and matching, and be careful normalizing text
# you send to a model whose tokenizer was trained on raw text.
#
# ## 5 · Find the token-length percentile for `max_length` (2')
#
# Before choosing `max_length` for a classifier, look at the length distribution of your training data
# (in tokens of the model's own tokenizer, special tokens included).
#
# ✏️ **Exercise** · Compute the 95th and 99th percentiles of `train_lengths` (one line with NumPy).

# %%
train_lengths = np.array([len(ids) for ids in tok(train.text.tolist())["input_ids"]])
ar_lengths = np.array([len(ids) for ids in tok(ar_texts)["input_ids"]])

# BEGIN SOLUTION
p95, p99 = np.percentile(train_lengths, [95, 99])
# END SOLUTION
print(f"train tickets (XLM-R tokens): p95 = {p95:.0f}, p99 = {p99:.0f}, max = {train_lengths.max()}")

# %%
p95_ar, p99_ar = np.percentile(ar_lengths, [95, 99])
max_length = int(np.ceil(max(np.percentile(train_lengths, 99), p99_ar) / 8) * 8)  # round up to a multiple of 8
print(f"Arabic tickets: p95 = {p95_ar:.0f}, p99 = {p99_ar:.0f}  ->  suggested max_length = {max_length}")
print(f"share of train tickets truncated at max_length={max_length}: {(train_lengths > max_length).mean():.2%}")

fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.arange(0, max(train_lengths.max(), ar_lengths.max()) + 2)
ax.hist(train_lengths, bins=bins, alpha=0.6, density=True, label="EN train")
ax.hist(ar_lengths, bins=bins, alpha=0.6, density=True, label="AR test")
ax.axvline(max_length, color="black", ls="--", label=f"max_length = {max_length}")
ax.set_xlabel("tokens per ticket (XLM-R, with special tokens)")
ax.set_ylabel("share of tickets")
ax.set_title("Ticket length in tokens")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Our tickets are short chat messages, so a small `max_length` is enough and keeps training and inference cheap.
# Real e-mails are longer: always check, and remember that the question is often at the *end* of a long e-mail,
# which plain truncation cuts off.
#
# ## ✱ Stretch · Train a tiny BPE tokenizer on the tickets
#
# The `tokenizers` library (the fast Rust backend of `transformers`) can learn a BPE vocabulary in seconds.
# We train on all English training tickets plus the 108 Arabic tickets, so Arabic is heavily under-represented,
# just like in GPT-2's training data.

# %%
import json

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

corpus = train.text.tolist() + ar_texts
bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=U.pick(2000, 1000), min_frequency=2, special_tokens=["[UNK]", "[PAD]"])
bpe.train_from_iterator(corpus, trainer)

merges = json.loads(bpe.to_str())["model"]["merges"]
print("vocab size:", bpe.get_vocab_size(), "| merges learned:", len(merges))
print("first 15 merges:", [" + ".join(m) if isinstance(m, list) else m for m in merges[:15]])
print("last 10 merges :", [" + ".join(m) if isinstance(m, list) else m for m in merges[-10:]])

# %%
for text in ["I want a refund for order A-12345", "how do I change my shipping address",
             "أين طلبي؟ لم يصل بعد", "أريد استرداد المبلغ"]:
    print(f"{len(bpe.encode(text).tokens):3d}  {bpe.encode(text).tokens}")

class _BPEWrapper:  # make the tiny tokenizer usable with fertility()
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": bpe.encode(text).ids}

print(f"tiny BPE fertility  EN: {fertility(_BPEWrapper(), en_texts):.2f} | AR: {fertility(_BPEWrapper(), ar_texts):.2f}")

# %% [markdown]
# 🔍 **Look at** the first merges (frequent English letter pairs) and how Arabic words stay split into
# letters or short pieces: the vocabulary mirrors the training corpus. Try adding more Arabic text to `corpus`
# (e.g. repeat `ar_texts` 50 times) and re-train: Arabic fertility drops.
#
# ## Checkpoint
#
# 1. A 60-word Arabic ticket goes through GPT-2's tokenizer. Roughly how many tokens do you expect, and why are there no `[UNK]`s?
# 2. Why should you not normalize alef forms in text you send to an LLM, but still do it for a search index?
# 3. You set `max_length=16` for a classifier on these tickets. What goes wrong, and how would you detect it?
#
# <details><summary>Answers</summary>
#
# 1. Around 5 tokens per word, so roughly 300 tokens (see the fertility table). GPT-2 uses byte-level BPE: every byte is in the
#    vocabulary, so nothing is unknown; Arabic simply gets few merges and splits into short byte pieces.
# 2. The LLM's tokenizer and model were trained on raw text, so normalized spellings can be rarer strings (higher fertility,
#    less natural input). For search, folding variants together increases matches between query and index.
# 3. A visible share of tickets (especially Arabic ones) gets truncated and the model never sees the end of the message.
#    Detect it by computing the token-length distribution (p95/p99) with the model's own tokenizer, per language.
#
# </details>
#
# ## Carry forward
#
# - Lab 1.2 embeds tickets with `multilingual-e5-small`, whose tokenizer is the XLM-R SentencePiece vocabulary you inspected here.
# - `normalize_ar` and the length analysis come back when we build search (Day 2 RAG) and fine-tune an encoder (Day 3):
#   choose `max_length` from data, per language. Token counts, not word counts, drive the LLM cost comparison on Day 2.
