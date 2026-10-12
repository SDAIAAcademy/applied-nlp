# %% [markdown]
# # Lab 1.3 · Open the black box
#
# Day 1 · Session 3 (Inside the transformer) · 25 min. What you'll build:
# - scaled dot-product attention in a few lines of NumPy, checked against PyTorch;
# - a look inside multilingual BERT: hidden states, attention maps, and how context changes the vector of "charge";
# - masked-language-model predictions in English and Arabic with XLM-R.
#
# **Tasks**
# - [ ] 1 · Implement scaled dot-product attention in 10 lines of NumPy (6')
# - [ ] 2 · Inspect hidden-state and attention shapes for a ticket (4')
# - [ ] 3 · Plot an attention heatmap for a chosen layer/head (5')
# - [ ] 4 · Contextual embeddings: "charge" in two contexts (5')
# - [ ] 5 · Fill-mask in English and Arabic with XLM-R (5')
# - [ ] ✱ Stretch · Causal mask: compare encoder vs. decoder attention patterns

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1 · Implement scaled dot-product attention in 10 lines of NumPy (6')
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$
#
# - Row *i* of `Q @ K.T` holds the scores of query token *i* against every key token.
# - Dividing by √d keeps the scores in a range where softmax is not saturated.
# - Softmax turns each row into weights that sum to 1; the output is the weighted sum of the value vectors.
# - An optional boolean `mask` (True = may attend) blocks positions: padding, or the future in a decoder.
#   Blocked scores are set to a very negative number before the softmax, so their weight becomes 0.
#
# ✏️ **Exercise** · Complete `attention(Q, K, V, mask=None)`. Return both the output and the weight matrix.

# %%
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)    # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """Q: (n_q, d), K: (n_k, d), V: (n_k, d_v), mask: (n_q, n_k) bool or None -> (output (n_q, d_v), weights (n_q, n_k))"""
    # BEGIN SOLUTION
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
    # END SOLUTION

# %%
# Check + safety net
rng = np.random.default_rng(0)
n, d = 5, 8
Q, K, V = rng.normal(size=(n, d)), rng.normal(size=(n, d)), rng.normal(size=(n, d))
try:
    out, w = attention(Q, K, V)
except NotImplementedError:
    print("Exercise skipped: using the reference attention()")
    def attention(Q, K, V, mask=None):
        scores = Q @ K.T / np.sqrt(Q.shape[-1])
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        weights = softmax(scores)
        return weights @ V, weights
    out, w = attention(Q, K, V)

print("output:", out.shape, "| weights:", w.shape)
print("rows sum to 1:", np.allclose(w.sum(axis=1), 1))
ref = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(a)[None] for a in (Q, K, V)))[0].numpy()
print("matches torch.nn.functional.scaled_dot_product_attention:", np.allclose(out, ref, atol=1e-6))
print(w)

# %% [markdown]
# A real layer first projects each token vector into Q, K and V with learned matrices and runs this
# in parallel for 12 heads. That is the whole trick; everything else is stacking.
#
# ## 2 · Inspect hidden-state and attention shapes for a ticket (4')
#
# We load multilingual BERT (encoder-only, 12 layers, 12 heads, hidden size 768) and ask it to return all
# hidden states and all attention weights (code from the slide "Hugging Face mental model").
#
# > Asking for attentions makes `transformers` use the plain ("eager") attention implementation instead of the fused one,
# > because fused kernels never materialize the weight matrix. Fine for inspection, slower for production.

# %%
MBERT = "google-bert/bert-base-multilingual-cased"   # same repo as "bert-base-multilingual-cased"
tok = AutoTokenizer.from_pretrained(MBERT)
model = AutoModel.from_pretrained(MBERT, output_attentions=True, output_hidden_states=True).eval()

ticket = "You charged me twice for one order"
enc = tok(ticket, return_tensors="pt")
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
with torch.no_grad():
    out = model(**enc)

print("tokens:", tokens)
print("last_hidden_state:", tuple(out.last_hidden_state.shape), " (batch, seq_len, hidden)")
print("hidden_states    :", len(out.hidden_states), "tensors (embeddings + 12 layers), each", tuple(out.hidden_states[0].shape))
print("attentions       :", len(out.attentions), "tensors (one per layer), each", tuple(out.attentions[0].shape), " (batch, heads, query, key)")
print("parameters       :", f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

# %% [markdown]
# 🔍 **Look at** where the parameters are: most of the 178M sit in the 119k × 768 embedding matrix
# (a large multilingual vocabulary), not in the 12 transformer layers.

# %%
emb_params = model.embeddings.word_embeddings.weight.numel()
print(f"word embedding matrix: {tuple(model.embeddings.word_embeddings.weight.shape)} = {emb_params / 1e6:.0f}M parameters")

# %% [markdown]
# ## 3 · Plot an attention heatmap for a chosen layer/head (5')
#
# Rows are the tokens that attend (queries), columns the tokens attended to (keys). Each row sums to 1.
# Change `LAYER` and `HEAD` and look for patterns: attend-to-previous-token, attend-to-`[SEP]`, attend-to-self,
# or content heads that link "charged" with "twice" and "order".

# %%
LAYER, HEAD = 8, 9   # 0-based; try others

def plot_attention(att, tokens, title, ax=None):
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]
    im = ax.imshow(att, cmap="Blues", vmin=0)   # colour scale per map
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=8)
    ax.set_xlabel("attended token (key)")
    ax.set_ylabel("attending token (query)")
    ax.set_title(title, fontsize=9)
    return im

att = out.attentions[LAYER][0, HEAD].numpy()
im = plot_attention(att, tokens, f"mBERT attention, layer {LAYER}, head {HEAD}")
plt.colorbar(im, fraction=0.046)
plt.tight_layout()
plt.show()

i = tokens.index("charged")
print("'charged' attends most to:", [(tokens[j], round(float(att[i, j]), 2)) for j in np.argsort(-att[i])[:3]])

# %% [markdown]
# Multi-head attention: the 12 heads of the same layer side by side. Each learns a different pattern.

# %%
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for h, ax in enumerate(axes.ravel()):
    plot_attention(out.attentions[LAYER][0, h].numpy(), tokens, f"layer {LAYER}, head {h}", ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle(f"All heads of layer {LAYER} (rows: query token, columns: key token)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Many heads put most of their weight on `[SEP]` or `[CLS]`: a "no-op" pattern. This is why attention maps are a
# diagnostic, not an explanation of the model's decision.
#
# ## 4 · Contextual embeddings: "charge" in two contexts (5')
#
# We take the vector of the *same word* at every layer (embedding output + 12 layers) in sentences with different meanings
# and compare them with cosine similarity.
#
# Subword handling: a word may be split into several tokens ("charger" → `charge` + `##r`), and the token index differs per
# sentence. We use the tokenizer's **offset mapping** (the character span of each token) to find the tokens that belong to
# the word, and average their vectors.

# %%
for s in ["You charged me twice for one order", "The charger stopped working"]:
    e = tok(s, return_offsets_mapping=True)
    print(list(zip(tok.convert_ids_to_tokens(e["input_ids"]), e["offset_mapping"])))

# %% [markdown]
# ✏️ **Exercise** · In `word_vectors`, build `idx`: the positions of the tokens whose character span `(a, b)` lies inside
# `[start, end)` of the word. Skip special tokens, which have the empty span `(0, 0)`.

# %%
def word_vectors(sentence, word):
    """Vector of `word` (first occurrence) at every layer: array (n_layers + 1, hidden)."""
    enc = tok(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    start = sentence.index(word)
    end = start + len(word)
    # BEGIN SOLUTION
    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
    # END SOLUTION
    with torch.no_grad():
        hidden_states = model(**enc).hidden_states
    return torch.stack([h[0, idx].mean(dim=0) for h in hidden_states]).numpy()

# %%
# Check + safety net
try:
    assert word_vectors("The charger stopped working", "charger").shape == (13, 768)
    print("word_vectors() looks right")
except NotImplementedError:
    print("Exercise skipped: using the reference word_vectors()")
    def word_vectors(sentence, word):
        enc = tok(sentence, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        start = sentence.index(word); end = start + len(word)
        idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
        with torch.no_grad():
            hidden_states = model(**enc).hidden_states
        return torch.stack([h[0, idx].mean(dim=0) for h in hidden_states]).numpy()

# %%
examples = {
    "billing A": ("You charged me twice for one order", "charged"),
    "billing B": ("There is a charge I don't recognize on my card", "charge"),
    "device A":  ("My phone won't charge since the update", "charge"),
    "device B":  ("The battery does not charge anymore", "charge"),
}
vecs = {k: word_vectors(s, w) for k, (s, w) in examples.items()}

def layer_cosine(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

pairs = [("device A", "device B", "same sense, same token"),
         ("device A", "billing B", "different sense, same token"),
         ("billing A", "billing B", "same sense, charged vs. charge"),
         ("billing A", "device A", "slide pair: charged vs. charge")]
cos_table = pd.DataFrame({f"{a} vs {b}": layer_cosine(vecs[a], vecs[b]) for a, b, _ in pairs})
cos_table.index.name = "layer"
cos_table.round(2).T

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for (a, b, desc), col in zip(pairs, cos_table.columns):
    ax.plot(cos_table.index, cos_table[col], marker="o", label=f"{col} ({desc})")
ax.set_xlabel("layer (0 = embedding output)")
ax.set_ylabel("cosine similarity")
ax.set_title('Vector of "charge" across layers, mBERT')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at** the two "same token" lines. At layer 0 the vectors are almost identical (only the position differs).
# Layer by layer, context pulls them apart: the two *device* uses stay close, while *device* vs. *billing* drifts away.
# That is contextualization. The "charged" vs. "charge" pairs start far apart because they are different tokens, and
# mBERT does not clearly group them by sense: a general-purpose model is not a sense classifier.
# The last layer is specialised for the pretraining task, which is why middle layers often work best as features.
#
# ## 5 · Fill-mask in English and Arabic with XLM-R (5')
#
# Encoders are pretrained by masked language modeling: hide a token, predict it from both sides.
# The `fill-mask` pipeline shows the model's top guesses. XLM-R's mask token is `<mask>`.

# %%
fill = pipeline("fill-mask", model="FacebookAI/xlm-roberta-base")

prompts = [
    "My <mask> has not arrived yet.",
    "My <mask> has not arrived yet, I ordered it last week.",
    "لم يصل <mask> الذي طلبته.",
    "نسيت كلمة <mask> ولا أستطيع الدخول إلى حسابي.",
]
rows = []
for p in prompts:
    preds = fill(p, top_k=5)
    rows.append({"prompt": p, **{f"#{k + 1}": f"{r['token_str']} ({r['score']:.2f})" for k, r in enumerate(preds)}})
pd.DataFrame(rows).set_index("prompt")

# %% [markdown]
# 🔍 **Look at**
# - "My `<mask>` has not arrived yet" → *email* beats *order*: without shop context, mail is the most common thing that "arrives".
#   Add context ("I ordered it last week") and *order* wins. Context is the whole game.
# - The Arabic "password" prompt is easy (كلمة المرور is a fixed expression). The other Arabic prompt is weak: the top guesses are
#   function words such as إلى. The mask is one SentencePiece token, but many Arabic words are several pieces (طلب + ي),
#   and XLM-R saw far less Arabic than English in pretraining. Try your own prompts: you will also see clitic fragments (ي, ه).
# - XLM-R was never trained on support tickets. It is a raw pretrained body; on Day 3 we fine-tune such a body with a task head.
#
# ## ✱ Stretch · Causal mask: compare encoder vs. decoder attention patterns
#
# A decoder (GPT, Qwen, Llama) may only look at the past: position *i* attends to positions ≤ *i*.
# With your NumPy `attention`, that is a lower-triangular mask.

# %%
causal = np.tril(np.ones((n, n), dtype=bool))
_, w_causal = attention(Q, K, V, mask=causal)
print(w_causal)
print("upper triangle is zero:", np.allclose(np.triu(w_causal, k=1), 0), "| rows sum to 1:", np.allclose(w_causal.sum(1), 1))

# %% [markdown]
# Now the same sentence through GPT-2 (decoder) and mBERT (encoder), attention averaged over the heads of one layer.

# %%
gpt_tok = AutoTokenizer.from_pretrained("gpt2")
gpt = AutoModel.from_pretrained("gpt2", output_attentions=True).eval()
g_enc = gpt_tok(ticket, return_tensors="pt")
with torch.no_grad():
    g_out = gpt(**g_enc)
g_tokens = gpt_tok.convert_ids_to_tokens(g_enc["input_ids"][0])

L = 5
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plot_attention(out.attentions[L][0].mean(0).numpy(), tokens, f"mBERT (encoder), layer {L}, mean over heads", ax=axes[0])
plot_attention(g_out.attentions[L][0].mean(0).numpy(), g_tokens, f"GPT-2 (decoder), layer {L}, mean over heads", ax=axes[1])
plt.tight_layout()
plt.show()
print("GPT-2 weight above the diagonal:", float(np.triu(g_out.attentions[L][0].mean(0).numpy(), k=1).sum()))

# %% [markdown]
# The encoder map is full: every token sees left and right context, which suits classification and embeddings.
# The decoder map is lower-triangular, which is what makes next-token generation possible (Day 2).
# Note also GPT-2's heavy weight on the first token, a common "attention sink" in decoders.
#
# ## Checkpoint
#
# 1. In your NumPy attention, what are the shapes of the score matrix and of the output for 7 tokens with d = 64?
#    Why does cost grow with the square of the sequence length?
# 2. Why is the layer-0 vector of "charge" (almost) the same in every sentence, and what changes it in later layers?
# 3. You need to route 2 million tickets per day into 27 intents. Encoder or decoder, and why?
#
# <details><summary>Answers</summary>
#
# 1. Scores: 7 × 7, output: 7 × 64. Every token scores every other token, so n tokens need n² scores (and memory).
# 2. Layer 0 is the token embedding plus a position embedding: no context yet. Each attention layer mixes in information from
#    the other tokens, so the vector becomes context-dependent.
# 3. An encoder with a classification head: the output is a label, volume is high and latency matters. Small encoders are cheap
#    and accurate once fine-tuned (Day 3).
#
# </details>
#
# ## Carry forward
#
# - Lab 1.4 uses an encoder in two ways: frozen (e5 embeddings + logistic regression) and as a zero-shot NLI model.
# - Day 2 works with decoders (causal attention, next-token prediction); Day 3 fine-tunes an encoder body like the one you opened here.
