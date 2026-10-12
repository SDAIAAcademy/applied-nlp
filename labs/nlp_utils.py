"""Shared plumbing for the Applied NLP labs.

Concepts are taught in the notebooks themselves; this module only holds the
boring, repeated parts: paths, data loading, model names, an LLM wrapper that
works with local Hugging Face models or any OpenAI-compatible endpoint,
embedding caching, metrics helpers and prediction files for the leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
ARTIFACTS = LABS / "artifacts"
CACHE = LABS / "cache"
PRED_DIR = ARTIFACTS / "predictions"

# Quick mode shrinks every workload (useful on CPU-only laptops and for CI).
QUICK = os.getenv("APPLIED_NLP_QUICK", "0") == "1"

try:  # optional .env with LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
    from dotenv import load_dotenv

    load_dotenv(LABS / ".env")
except ImportError:
    pass

MODELS = {
    "embed": "intfloat/multilingual-e5-small",
    "nli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "encoder": "distilbert/distilbert-base-multilingual-cased",
    "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "llm_gpu": "Qwen/Qwen2.5-1.5B-Instruct",
    "llm_cpu": "Qwen/Qwen2.5-0.5B-Instruct",
    "lora_base": "Qwen/Qwen2.5-0.5B-Instruct",
}


def pick(full, quick):
    """Return `quick` in quick mode, otherwise `full`."""
    return quick if QUICK else full


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python prepare_data.py` in the labs folder first.")
    return path


def load_tickets(split: str) -> pd.DataFrame:
    """split: train | val | test | history | ar_test"""
    return pd.read_csv(_require(DATA / f"tickets_{split}.csv"))


def load_labels() -> dict:
    return json.loads(_require(DATA / "labels.json").read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_kb() -> list[dict]:
    """Help-center articles split by `## ` headings.

    Returns one dict per section: {id, doc, title, section, text}. The id has the
    form "refunds#refund-timelines" and matches data/rag_eval.csv.
    """
    sections = []
    for path in sorted((DATA / "kb").glob("*.md")):
        doc = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].lstrip("# ").strip()
        current, buf = None, []
        for line in lines[1:] + ["## __end__"]:
            if line.startswith("## "):
                if current and "\n".join(buf).strip():
                    sections.append({
                        "id": f"{doc}#{slugify(current)}",
                        "doc": doc,
                        "title": title,
                        "section": current,
                        "text": "\n".join(buf).strip(),
                    })
                current, buf = line[3:].strip(), []
            elif current:
                buf.append(line)
    return sections


# ---------------------------------------------------------------------------
# Embeddings with an on-disk cache (several labs embed the same splits)
# ---------------------------------------------------------------------------
_EMBEDDERS: dict = {}


def get_embedder(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    name = name or MODELS["embed"]
    if name not in _EMBEDDERS:
        _EMBEDDERS[name] = SentenceTransformer(name, device=device())
    return _EMBEDDERS[name]


def embed(texts: Iterable[str], prefix: str = "query: ", model: str | None = None,
          batch_size: int = 64, cache_key: str | None = None) -> np.ndarray:
    """Normalized sentence embeddings. e5 models expect 'query: ' / 'passage: ' prefixes."""
    texts = list(texts)
    model = model or MODELS["embed"]
    if cache_key:
        CACHE.mkdir(exist_ok=True)
        digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:10]  # same key, other texts -> new file
        f = CACHE / f"{slugify(model)}__{cache_key}__{slugify(prefix) or 'noprefix'}__{len(texts)}_{digest}.npy"
        if f.exists():
            return np.load(f)
    vecs = get_embedder(model).encode(
        [prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=len(texts) > 500, convert_to_numpy=True,
    ).astype("float32")
    if cache_key:
        np.save(f, vecs)
    return vecs


# ---------------------------------------------------------------------------
# LLM wrapper: local Hugging Face model or OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
@dataclass
class LLM:
    """llm = LLM(); llm.chat([{"role": "user", "content": "hi"}])

    Backend selection: if LLM_BASE_URL is set, use that OpenAI-compatible endpoint
    (vLLM, Ollama, a hosted provider...). Otherwise load a local Qwen2.5 model:
    1.5B on GPU, 0.5B on CPU. Pass `model=` to override.
    """

    model: str | None = None
    backend: str | None = None
    last_usage: dict = field(default_factory=dict)

    def __post_init__(self):
        self.backend = self.backend or ("openai" if os.getenv("LLM_BASE_URL") else "hf")
        if self.backend == "openai":
            from openai import OpenAI

            self.model = self.model or os.environ["LLM_MODEL"]
            self.client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.getenv("LLM_API_KEY", "none"))
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dev = device()
            self.model = self.model or (MODELS["llm_gpu"] if dev == "cuda" else MODELS["llm_cpu"])
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.hf_model = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype).to(dev)
            self.hf_model.eval()
        print(f"LLM ready: {self.model} via {self.backend}")

    def chat(self, messages: list[dict], temperature: float = 0.0, max_new_tokens: int = 256,
             top_p: float = 1.0, **kwargs) -> str:
        t0 = time.perf_counter()
        if self.backend == "openai":
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature,
                max_tokens=max_new_tokens, top_p=top_p, **kwargs,
            )
            text = resp.choices[0].message.content or ""
            u = resp.usage
            self.last_usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                               "completion_tokens": getattr(u, "completion_tokens", None)}
        else:
            import torch

            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(self.hf_model.device)
            gen = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            if temperature and temperature > 0:
                gen.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            gen.update(kwargs)
            with torch.no_grad():
                out = self.hf_model.generate(**enc, **gen)
            new = out[0][enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new, skip_special_tokens=True)
            self.last_usage = {"prompt_tokens": int(enc["input_ids"].shape[1]), "completion_tokens": int(len(new))}
        self.last_usage["latency_s"] = time.perf_counter() - t0
        return text.strip()


_LLM: LLM | None = None


def get_llm(**kwargs) -> LLM:
    """Load the LLM once per kernel."""
    global _LLM
    if _LLM is None or kwargs:
        _LLM = LLM(**kwargs)
    return _LLM


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, labels=None, title="Confusion matrix", figsize=(9, 8), normalize=True):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def top_confusions(y_true, y_pred, k: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"true": list(y_true), "pred": list(y_pred)})
    df = df[df.true != df.pred]
    return df.value_counts().head(k).rename("count").reset_index()


def bootstrap_ci(y_true, y_pred, metric, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for any metric(y_true, y_pred)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_pred)), float(lo), float(hi)


def bench(fn, inputs: list, warmup: int = 3) -> dict:
    """Latency percentiles (ms) of fn(x) over inputs, after warm-up calls."""
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"n": len(t), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "per_sec": float(1000 / t.mean())}


# ---------------------------------------------------------------------------
# Prediction files: every lab that evaluates a model on the test sets saves one,
# and Lab 3.4 turns them into a single leaderboard.
# ---------------------------------------------------------------------------
def save_predictions(name: str, df: pd.DataFrame, meta: dict | None = None) -> Path:
    """df needs columns: ticket_id, lang, y_true, y_pred (optional: confidence)."""
    missing = {"ticket_id", "lang", "y_true", "y_pred"} - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{slugify(name)}.csv"
    df.to_csv(path, index=False)
    (PRED_DIR / f"{slugify(name)}.json").write_text(
        json.dumps({"name": name, **(meta or {})}, indent=2), encoding="utf-8")
    print(f"Saved {len(df)} predictions -> {path.relative_to(LABS)}")
    return path


def load_predictions() -> dict[str, tuple[pd.DataFrame, dict]]:
    out = {}
    for csv in sorted(PRED_DIR.glob("*.csv")):
        meta_file = csv.with_suffix(".json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {"name": csv.stem}
        out[meta.get("name", csv.stem)] = (pd.read_csv(csv), meta)
    return out
