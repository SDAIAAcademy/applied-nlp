# %% [markdown]
# # Lab 0 · Setup check
#
# Run this notebook **before Day 1** (it takes 5–15 minutes, mostly downloads). It:
#
# - installs packages on Colab and prepares the course data,
# - checks your Python packages and whether a GPU is available,
# - downloads every model used in the labs into the local Hugging Face cache, so class time isn't spent waiting,
# - runs a 30-second smoke test of embeddings and the LLM.
#
# On Colab: **Runtime → Change runtime type → T4 GPU** first. Locally: `cd labs && uv sync && uv run jupyter lab`.

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

# %% [markdown]
# ## 1 · Packages and hardware

# %%
import importlib
import platform

packages = ["torch", "transformers", "tokenizers", "datasets", "accelerate", "peft",
            "sentence_transformers", "faiss", "rank_bm25", "sklearn", "pandas", "pydantic",
            "openai", "gradio"]
rows = []
for name in packages:
    try:
        mod = importlib.import_module(name)
        rows.append((name, getattr(mod, "__version__", "ok")))
    except ImportError as e:
        rows.append((name, f"MISSING ({e.__class__.__name__})"))

import pandas as pd
import torch

print("Python", platform.python_version(), "|", platform.system())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
else:
    print("No CUDA GPU: Days 1-2 run fine on CPU; use Colab with a T4 for Labs 3.2 and 3.3.")
pd.DataFrame(rows, columns=["package", "version"])

# %% [markdown]
# ## 2 · Data
#
# `prepare_data.py` downloaded the Bitext customer-support dataset and wrote the splits below. Arabic tickets are hand-written (MSA and Gulf dialect).

# %%
summary = []
for split in ["train", "val", "test", "history", "ar_test"]:
    df = U.load_tickets(split)
    summary.append({"split": split, "rows": len(df), "intents": df.intent.nunique(), "lang": df.lang.iloc[0]})
print("KB sections:", len(U.load_kb()))
pd.DataFrame(summary)

# %% [markdown]
# ## 3 · Download every model used in the labs
#
# Already-downloaded models are skipped. Total is roughly 6 GB with the 1.5B LLM (GPU), less on CPU.

# %%
from huggingface_hub import snapshot_download

models = [
    U.MODELS["embed"], U.MODELS["nli"], U.MODELS["encoder"], U.MODELS["reranker"], U.MODELS["lora_base"],
    "gpt2", "bert-base-multilingual-cased", "xlm-roberta-base", "aubmindlab/bert-base-arabertv02",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
if torch.cuda.is_available():
    models.append(U.MODELS["llm_gpu"])

skip = ["*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite", "onnx/*", "openvino/*", "*.gguf", "coreml/*"]
failed = []
for m in models:
    for attempt in range(3):  # the Hub occasionally drops connections; retry
        try:
            snapshot_download(m, ignore_patterns=skip)
            print(f"ok    {m}")
            break
        except Exception as e:
            err = e
    else:
        failed.append(m)
        print(f"FAIL  {m}: {str(err)[:200]}")
print("All models downloaded." if not failed else f"Re-run this cell later for: {failed}")

# %% [markdown]
# ## 4 · Smoke test
#
# One embedding similarity and one LLM reply. If you set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `labs/.env`, the LLM call goes to that OpenAI-compatible endpoint instead of the local model.

# %%
docs = ["Where is my refund?", "How do I change my password?", "I want to cancel my order"]
queries = ["لم يصلني المبلغ المسترد حتى الآن", "كيف أغير كلمة المرور؟"]
D = U.embed(docs, prefix="passage: ")
Q = U.embed(queries, prefix="query: ")
for q, scores in zip(queries, Q @ D.T):
    print(f"{q}  ->  {docs[scores.argmax()]}   scores={scores.round(3)}")

llm = U.get_llm()
reply = llm.chat([{"role": "user", "content": "In one short sentence: what is a support ticket?"}], max_new_tokens=40)
print(reply)
print(llm.last_usage)

# %% [markdown]
# ## You're ready
#
# If every package shows a version, every model says `ok`, and the smoke test printed a reply, you're set for Day 1.

Each Arabic query should retrieve the matching English sentence: cross-lingual retrieval, which we use all week. Notice how close the scores are (all around 0.75–0.86). e5 similarities are compressed into a narrow range, so compare **rankings**, not absolute values.
