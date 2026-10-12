# Applied NLP Engineering: labs

Hands-on notebooks for the SDAIA Academy **Applied NLP Engineering** bootcamp (AI Engineering track). Every lab builds a piece of one system: an **intelligent support-ticket assistant** for English and Arabic customer messages.

## Quick start

**Google Colab (recommended for Day 3)**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/applied-nlp/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/<notebook>.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*, then run the first cell. It clones this repository, installs `requirements-colab.txt` and prepares the data.

**Local (CPU is fine for Days 1–2)**
```bash
git clone https://github.com/SDAIAAcademy/applied-nlp
cd applied-nlp/labs
uv sync                        # creates .venv from pyproject.toml
uv run python prepare_data.py  # downloads Bitext and writes data/*.csv
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1: it downloads every model so class time isn't spent waiting.

Set `APPLIED_NLP_QUICK=1` to shrink every workload (smaller samples, a few training steps). Useful on slow laptops and for checking that notebooks run.

## Notebooks

| Day | Session | Notebook | Runtime |
|---|---|---|---|
| 0 | Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/00_setup_check.ipynb) | CPU/GPU |
| 1 | Tokenization | [`D1_1_tokenization.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_1_tokenization.ipynb) | CPU |
| 1 | Embeddings & semantic search | [`D1_2_embeddings_semantic_search.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_2_embeddings_semantic_search.ipynb) | CPU |
| 1 | Inside the transformer | [`D1_3_transformers_inside.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_3_transformers_inside.ipynb) | CPU |
| 1 | Decision models | [`D1_4_decision_models.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D1_4_decision_models.ipynb) | CPU |
| 2 | Generative LLMs | [`D2_1_generative_llms.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_1_generative_llms.ipynb) | CPU (GPU faster) |
| 2 | Prompting & structured outputs | [`D2_2_prompting_structured_outputs.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_2_prompting_structured_outputs.ipynb) | CPU (GPU faster) |
| 2 | RAG | [`D2_3_rag.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_3_rag.ipynb) | CPU (GPU faster) |
| 2 | Model selection & evaluation | [`D2_4_model_selection_evaluation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D2_4_model_selection_evaluation.ipynb) | CPU (GPU faster) |
| 3 | Fine-tuning data prep | [`D3_1_finetuning_data_prep.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_1_finetuning_data_prep.ipynb) | CPU |
| 3 | Fine-tune an encoder | [`D3_2_finetune_encoder.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_2_finetune_encoder.ipynb) | **GPU** |
| 3 | LoRA for an LLM | [`D3_3_lora_llm.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_3_lora_llm.ipynb) | **GPU** |
| 3 | Evaluate & deploy | [`D3_4_evaluate_deploy.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/applied-nlp/blob/main/labs/D3_4_evaluate_deploy.ipynb) | CPU |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in. Complete reference versions are in `solutions/`. Cells marked **✱ Stretch** are optional.

Labs that evaluate an intent classifier save their test predictions to `artifacts/predictions/`; Lab 3.4 turns them into one leaderboard.

## Data

`prepare_data.py` builds everything in `data/`:

| File | Content |
|---|---|
| `tickets_train.csv` / `_val` / `_test` | Bitext customer-support messages (EN), 27 intents, 11 categories, 200 / 40 / 40 per intent |
| `tickets_history.csv` | 8,000 resolved tickets with agent responses (search and RAG index) |
| `tickets_ar_test.csv` | 108 hand-written Arabic tickets (MSA and Gulf dialect), same labels |
| `labels.json` | Label lists and the intent → category map |
| `kb/*.md` | Help-center articles of **DemoMart**, a fictional store (RAG knowledge base) |
| `rag_eval.csv` | Questions with the KB section that answers them (2 are unanswerable) |

`priority` (low / medium / high / urgent) is a business rule: a base level per intent, raised one level when the message shows urgency or anger. Source: [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (CDLA-Sharing-1.0). The Arabic tickets, KB and evaluation questions were written for this course.

## Using a stronger LLM

By default the labs run Qwen2.5-1.5B-Instruct on a GPU or Qwen2.5-0.5B-Instruct on CPU. To use any OpenAI-compatible endpoint (vLLM, Ollama, or a hosted provider), copy `.env.example` to `.env` and fill it in. Check your organisation's data policy before sending ticket text to an external service.

## Editing the notebooks

Notebooks are generated from `_src/*.py` (percent format). Edit the source, then run:
```bash
uv run python _tools/py2nb.py            # rebuild all (student + solutions)
uv run python _tools/py2nb.py D2_3_rag   # rebuild one
```
