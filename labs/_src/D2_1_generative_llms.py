# %% [markdown]
# # Lab 2.1 · Run and probe a small instruct model
#
# Day 2 · Session 1 (Generative LLMs) · 25 min
#
# **What you'll build**
# - A view of what the model really sees (chat template) and what it predicts next (a probability table).
# - Your own greedy decoding loop, checked against `model.generate`.
# - Latency numbers for prefill vs. decode, the two phases of every LLM call.
#
# **Tasks** (matching the lab slide)
# 1. Render the chat template; count prompt tokens (4')
# 2. Inspect the top-10 next-token probabilities (5')
# 3. ✏️ Write your own greedy loop with `model()` calls (6')
# 4. Compare greedy vs. T=0.7 vs. T=1.5 on a reply draft (×3 runs) (5')
# 5. Measure prefill vs. decode time; tokens per second (5')
# 6. ✱ Stretch: ask in Arabic and English; compare quality and token counts
#
# The model is **Qwen2.5-1.5B-Instruct on a GPU** and **Qwen2.5-0.5B-Instruct on CPU**. These are small models: fluent, but they make factual and formatting mistakes that a 70B model would not. That's fine for this lab: we are studying the *mechanics*, which are the same at every size.

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
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 120)

# %% [markdown]
# ### Load the model
#
# This lab needs the model's **weights and tokenizer** (we call `model(...)` directly), so we always load a local Hugging Face model with `U.LLM(backend="hf")`. It picks 1.5B on GPU and 0.5B on CPU and is equivalent to the slide code (`AutoTokenizer` + `AutoModelForCausalLM`).
#
# In the other Day 2 labs, `U.get_llm()` is used instead. If you put `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `labs/.env`, every `U.get_llm()` call switches to that OpenAI-compatible endpoint (vLLM, Ollama, a hosted provider), usually a much stronger model. This lab stays local because an API does not give you logits.

# %%
llm = U.LLM(backend="hf")          # local weights, whatever is in .env
tok, model = llm.tokenizer, llm.hf_model
DEV = model.device
print(model.config.model_type, f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", "| vocab:", len(tok))


def sync():
    """Wait for the GPU to finish before reading the clock (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# %% [markdown]
# ## 1 · Render the chat template; count prompt tokens (4')
#
# A chat model was fine-tuned on conversations written in one exact format. `apply_chat_template` turns a list of messages into that format. With `tokenize=False` you get the string, so you can read it.

# %%
messages = [
    {"role": "system", "content": "You are a support triage assistant."},
    {"role": "user", "content": "Where is my refund?"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

ids = tok(prompt, return_tensors="pt")["input_ids"][0]
content_tokens = sum(len(tok(m["content"])["input_ids"]) for m in messages)
print(f"prompt tokens: {len(ids)}  (message content: {content_tokens}, template overhead: {len(ids) - content_tokens})")
print(tok.convert_ids_to_tokens(ids))

# %% [markdown]
# Two details worth seeing with your own eyes:
#
# - `add_generation_prompt=True` appends `<|im_start|>assistant\n`: the cue that it's the model's turn. Without it, the model may continue the *user's* message.
# - If you send no system message, Qwen's template **inserts a default one**. Other model families do different things, which is why you always let the tokenizer apply the template.

# %%
no_system = tok.apply_chat_template(messages[1:], tokenize=False, add_generation_prompt=True)
no_gen = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print("--- no system message ---\n" + no_system)
print("--- add_generation_prompt=False ---\n" + no_gen)

# %% [markdown]
# 🔍 **Look at**: the special tokens `<|im_start|>` and `<|im_end|>` are single tokens. Every call pays the template overhead, on top of your instructions and the ticket.

# %% [markdown]
# ## 2 · Inspect the top-10 next-token probabilities (5')
#
# We start the assistant's answer ourselves with `"Your refund will be"` and ask the model for the distribution over the **next** token. One forward pass returns logits of shape `[batch, seq_len, vocab]`; the last position is the prediction for the next token.

# %%
support_msgs = [
    {"role": "system", "content": "You are a helpful support agent for DemoMart, an online store."},
    {"role": "user", "content": "I returned my headphones last week. When will I get my money back?"},
]
prefix = "Your refund will be"
text = tok.apply_chat_template(support_msgs, tokenize=False, add_generation_prompt=True) + prefix
enc = tok(text, return_tensors="pt").to(DEV)

with torch.no_grad():
    logits = model(**enc).logits[0, -1].float()       # next-token logits, shape [vocab]
print("logits shape:", tuple(logits.shape))

probs = torch.softmax(logits, dim=-1)
top = probs.topk(10)
top_df = pd.DataFrame({
    "token": [repr(tok.decode([i])) for i in top.indices.tolist()],
    "id": top.indices.tolist(),
    "prob": top.values.tolist(),
})
top_df["cumulative"] = top_df["prob"].cumsum()
top_df.round(3)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(top_df["token"][::-1], top_df["prob"][::-1], color="#1c355e")
ax.set_title(f'p(next token) after "{prefix} …"')
ax.set_xlabel("probability")
ax.set_ylabel("token")
fig.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at**: tokens start with a space (`' processed'`), because the tokenizer merges the space into the word. How much probability do the top 10 cover? The other ~150k tokens share the rest.
#
# ### Temperature reshapes this distribution
#
# Sampling with temperature T means `softmax(logits / T)`. T < 1 sharpens (the top token gets even more mass), T > 1 flattens (rare tokens get a real chance).
#
# ✏️ **Exercise**: implement `softmax_with_temperature(logits, T)`. One line.

# %%
def softmax_with_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    # BEGIN SOLUTION
    return torch.softmax(logits / T, dim=-1)
    # END SOLUTION


temps = [0.5, 1.0, 1.5]
idx = top.indices
x = np.arange(len(idx))
fig, ax = plt.subplots(figsize=(9, 4))
for j, (T, color) in enumerate(zip(temps, ["#00ae8d", "#1c355e", "#e96852"])):
    p = softmax_with_temperature(logits, T)
    ax.bar(x + (j - 1) * 0.27, p[idx].cpu().numpy(), width=0.27, color=color,
           label=f"T={T} (top-10 mass {p[idx].sum():.2f})")
ax.set_xticks(x, [tok.decode([i]) for i in idx.tolist()], rotation=45)
ax.set_title("The same next-token distribution at three temperatures")
ax.set_xlabel("candidate next token")
ax.set_ylabel("probability")
ax.legend()
fig.tight_layout()
plt.show()

for T in temps:
    p = softmax_with_temperature(logits, T)
    entropy = -(p * torch.log(p + 1e-12)).sum().item()
    print(f"T={T}: p(top-1)={p.max():.3f}  entropy={entropy:.2f} nats  mass outside top-10={1 - p[idx].sum():.3f}")

# %% [markdown]
# ## 3 · Write your own greedy loop with `model()` calls (6')
#
# `generate` hides a simple loop: run the model, take the most likely token, append it, repeat.
#
# ✏️ **Exercise**: complete `greedy_generate`. At each step call `model(ids)`, take the logits of the **last** position, pick the `argmax`, and append it to `ids` with `torch.cat`. Stop early if the model emits the end-of-turn token (`tok.eos_token_id`, which is `<|im_end|>` for Qwen).

# %%
N_STEPS = U.pick(30, 15)


def greedy_generate(ids: torch.Tensor, n_steps: int) -> torch.Tensor:
    """ids: [1, seq_len] prompt token ids. Returns prompt + generated ids."""
    for _ in range(n_steps):
        # BEGIN SOLUTION
        with torch.no_grad():
            next_logits = model(ids).logits[:, -1, :]
        next_id = next_logits.argmax(dim=-1, keepdim=True)   # [1, 1]
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tok.eos_token_id:
            break
        # END SOLUTION
    return ids

# %% [markdown]
# Now compare with `model.generate(..., do_sample=False)`. One catch: Qwen ships a `generation_config` with defaults (`repetition_penalty=1.1`, sampling on). Print it, and override the penalty so both loops implement plain greedy decoding.

# %%
print(model.generation_config)

prompt_ids = enc["input_ids"]      # the prompt from task 2, ending in "Your refund will be"

sync(); t0 = time.perf_counter()
ours = greedy_generate(prompt_ids, N_STEPS)
sync(); t_ours = time.perf_counter() - t0

t0 = time.perf_counter()
with torch.no_grad():
    ref = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), max_new_tokens=N_STEPS,
                         do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.eos_token_id)
sync(); t_ref = time.perf_counter() - t0

n = prompt_ids.shape[1]
ours_new, ref_new = ours[0, n:].tolist(), ref[0, n:].tolist()
k = min(len(ours_new), len(ref_new))
print("ours    :", repr(prefix + tok.decode(ours_new)))
print("generate:", repr(prefix + tok.decode(ref_new)))
print(f"identical first {k} tokens: {ours_new[:k] == ref_new[:k]}")
print(f"time: our loop {t_ours:.2f}s vs generate {t_ref:.2f}s")

# %% [markdown]
# 🔍 **Look at**: the tokens should be identical. Your loop is slower because it re-processes the **whole** sequence at every step. `generate` keeps a **KV cache**: keys and values of past tokens are stored, so each step only computes the new token. That's the "KV cache" card on the slides.

# %% [markdown]
# ## 4 · Greedy vs. T=0.7 vs. T=1.5 on a reply draft (5')
#
# Same prompt, three decoding settings, three runs each. We pass `top_k=0` and `top_p=1.0` to switch off Qwen's default truncation, so you see the pure effect of temperature.

# %%
reply_msgs = [
    {"role": "system", "content": "You are a DemoMart support agent. Write a short, polite reply in 2 sentences."},
    {"role": "user", "content": "My order A-48213 arrived damaged. I want a replacement."},
]
MAX_NEW = U.pick(80, 40)
rows = []
for T in [0.0, 0.7, 1.5]:
    for run in range(3):
        torch.manual_seed(run)
        no_truncation = {"top_p": 1.0, "top_k": 0} if T > 0 else {}
        reply = llm.chat(reply_msgs, temperature=T, max_new_tokens=MAX_NEW, **no_truncation)
        rows.append({"setting": "greedy" if T == 0 else f"T={T}", "run": run, "reply": reply})
runs = pd.DataFrame(rows)
display(runs)
print(runs.groupby("setting", sort=False)["reply"].nunique().rename("distinct replies out of 3"))

# %% [markdown]
# 🔍 **Look at**: greedy gives the same text three times (what you want for classification and JSON). T=0.7 varies the wording but stays on topic (fine for drafting replies). At T=1.5 with no top-k/top-p, rare tokens get sampled and the text derails, sometimes into other languages. Production chat settings combine a moderate temperature with top-p ≈ 0.9 for exactly this reason.

# %% [markdown]
# ## 5 · Measure prefill vs. decode time; tokens per second (5')
#
# Every call has two phases:
# - **prefill**: the whole prompt goes through the model in **one parallel forward pass**;
# - **decode**: output tokens are produced **one at a time**, each a small forward pass using the KV cache.
#
# We build a long prompt (a help-center article + a question) and time one forward pass: that's the prefill.

# %%
article = (U.DATA / "kb" / "refunds.md").read_text(encoding="utf-8")
long_msgs = [
    {"role": "system", "content": "Answer the customer using only this article:\n\n" + article},
    {"role": "user", "content": "How long does a refund to my Visa card take?"},
]
long_enc = tok.apply_chat_template(long_msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEV)
n_prompt = long_enc["input_ids"].shape[1]

with torch.no_grad():
    model(**long_enc)                        # warm-up
    sync(); t0 = time.perf_counter()
    model(**long_enc)                        # prefill = one forward pass over the prompt
    sync(); prefill_s = time.perf_counter() - t0
print(f"prompt tokens: {n_prompt} | prefill: {prefill_s * 1000:.0f} ms | {n_prompt / prefill_s:.0f} prompt tokens/s")

# %% [markdown]
# ✏️ **Exercise**: measure the **decode** speed. Call `model.generate(**long_enc, ...)` greedily with `max_new_tokens=N_NEW` **and** `min_new_tokens=N_NEW` (so it can't stop early), time it, subtract `prefill_s`, and divide by `N_NEW` to get `decode_s_per_token`.

# %%
N_NEW = U.pick(64, 24)
# BEGIN SOLUTION
sync(); t0 = time.perf_counter()
with torch.no_grad():
    model.generate(**long_enc, max_new_tokens=N_NEW, min_new_tokens=N_NEW, do_sample=False,
                   temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
sync(); total_s = time.perf_counter() - t0
decode_s_per_token = (total_s - prefill_s) / N_NEW
# END SOLUTION

speed = pd.DataFrame({
    "phase": ["prefill (prompt)", "decode (output)"],
    "tokens": [n_prompt, N_NEW],
    "ms per token": [prefill_s / n_prompt * 1000, decode_s_per_token * 1000],
    "tokens per second": [n_prompt / prefill_s, 1 / decode_s_per_token],
})
display(speed.round(1))

# Estimated latency of two typical calls with the same prompt
tasks = {"classify (5 output tokens)": 5, "JSON (60 tokens)": 60, "reply draft (150 tokens)": 150}
est = pd.DataFrame({"prefill_s": prefill_s, "decode_s": [n * decode_s_per_token for n in tasks.values()]},
                   index=list(tasks.keys()))
ax = est.plot.barh(stacked=True, color=["#1c355e", "#e96852"], figsize=(7, 3))
ax.set_title(f"Estimated latency with a {n_prompt}-token prompt")
ax.set_xlabel("seconds")
ax.set_ylabel("task")
plt.tight_layout()
plt.show()

# %% [markdown]
# 🔍 **Look at**: per token, prefill is many times faster than decode (parallel vs. sequential). So **output length drives latency**: a classifier answering with a 5-token label is fast, a 150-token reply is slow. Capping `max_new_tokens` and asking for short outputs is the cheapest speed-up there is. On a GPU both phases are much faster, but the ratio stays.

# %% [markdown]
# ## ✱ Stretch · Arabic vs. English
#
# Ask the same question in both languages, with the same short policy in the system prompt. Compare the answers and the token counts. Recall Day 1: Arabic usually needs more tokens per word, which means more cost and more decode time for the same content.

# %%
def policy_msgs(q: str) -> list[dict]:
    return [{"role": "system", "content": "You are a DemoMart support agent. Policy: refunds to mada, Visa and "
                                          "Mastercard cards take 5 to 7 business days after the returned item is "
                                          "inspected. Answer in the customer's language in at most 2 sentences."},
            {"role": "user", "content": q}]

questions = {"en": "How long does a refund to my card take?",
             "ar": "كم يستغرق استرداد المبلغ إلى بطاقتي؟"}
rows = []
for lang, q in questions.items():
    answer = llm.chat(policy_msgs(q), max_new_tokens=U.pick(120, 60))
    rows.append({"lang": lang, "question tokens": len(tok(q)["input_ids"]), "question chars": len(q),
                 "prompt tokens": llm.last_usage["prompt_tokens"],
                 "answer tokens": llm.last_usage["completion_tokens"],
                 "answer chars": len(answer), "seconds": round(llm.last_usage["latency_s"], 1), "answer": answer})
pd.DataFrame(rows)

# %% [markdown]
# 🔍 **Look at**: did both answers state "5 to 7 business days"? Is the Arabic answer actually in Arabic, and is it fluent? Compare answer tokens per character. Small models are noticeably weaker in Arabic (and weaker still in dialects); this is one reason to evaluate on *your* language slice before choosing a model.

# %% [markdown]
# ## Checkpoint
#
# 1. Why does your greedy loop get slower with every step while `generate` does not?
# 2. You need a JSON triage result. Which decoding settings do you use, and why?
# 3. Your prompt is 900 tokens and the answer 60 tokens. Which phase dominates latency on a GPU, and what's the cheapest way to cut it?
#
# <details><summary>Answers</summary>
#
# 1. It re-runs the model on the whole sequence at each step (quadratic work). `generate` uses the KV cache, so each step only processes the new token.
# 2. Greedy (`do_sample=False` / temperature 0) and a small `max_new_tokens`: reproducible, most likely output, no random label flips.
# 3. Usually decode: 60 sequential steps vs. one parallel prefill pass. Ask for shorter outputs (compact JSON, labels only) and cap `max_new_tokens`; trimming the prompt helps cost and prefill too.
#
# </details>
#
# ## Carry forward
#
# Lab 2.2 uses the same model through `U.get_llm()` with greedy decoding to produce validated JSON. The token counts and prefill/decode numbers you measured here come back in Lab 2.4, where you estimate latency and cost per 1k tickets.
