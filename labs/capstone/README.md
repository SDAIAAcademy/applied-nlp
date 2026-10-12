# Capstone starter kit

Teams of 2–3 build an applied NLP solution and present it on Day 4 (10 min talk + live demo, 5 min Q&A).

## Choose a use case

| Use case | Builds on | Data ideas |
|---|---|---|
| **1 · Intelligent Support Ticket Assistant**: classify tickets, retrieve similar cases, draft grounded replies | Everything in the labs; `pipeline.py` is a working start | The course data (`labs/data`), plus your own Arabic tickets |
| **2 · Arabic Document Intelligence**: extract and categorize information from Arabic documents; answer questions with evidence | Labs 1.2, 1.4, 2.2, 2.3, 3.2 | Public Arabic news/classification corpora, or documents your organisation allows you to use |
| **3 · Voice of Customer Analyzer**: classify feedback, cluster issues with embeddings, generate insight summaries | Labs 1.2, 1.4, 2.2, 2.4, 3.2 | Public Arabic review datasets (e.g., hotel or book reviews), app-store reviews you may use |

Only use data you're allowed to use. Remove or mask personal data before sending anything to an external LLM API.

## Minimum requirements

1. A working end-to-end pipeline with a demo (Gradio or notebook UI).
2. At least one representation component (embeddings) and one decision or generation component.
3. A baseline and at least one improvement, compared on the same held-out test set.
4. Fine-tuning (encoder or LoRA), **or** a written, evidence-based justification for not fine-tuning.
5. Quantitative evaluation by slice (e.g., Arabic vs. English) + error analysis with ≥ 10 categorized failures.

## Rubric

| Criterion | Weight |
|---|---|
| Functional end-to-end NLP solution | 30% |
| Appropriate model selection and technical implementation | 25% |
| Quantitative evaluation and error analysis | 20% |
| Fine-tuning or justified model adaptation | 15% |
| Presentation and reproducibility | 10% |

## What's in this folder

| File | Purpose |
|---|---|
| `pipeline.py` | Working baseline: e5 + logistic regression classifier with abstention, rule-based category/priority, retrieval of similar tickets and KB sections, optional grounded LLM reply. Run `uv run python capstone/pipeline.py`. |
| `app.py` | Gradio demo around the pipeline: `uv run python capstone/app.py` |
| `evaluate.py` | Metrics with bootstrap CIs and per-slice tables from a predictions CSV: `uv run python capstone/evaluate.py preds.csv --slice lang` |
| `TEMPLATE_README.md` | README skeleton for your team repository |

## Known weaknesses of the baseline (good places to start)

- Bitext training data is templated: English test scores near 0.99 macro-F1 overstate real-world quality. Real phrasing breaks it: "I was charged twice for order A-55821, fix it now!" is routed to `change_order`.
- Arabic macro-F1 is roughly half the English score (cross-lingual transfer from English-only training), so most Arabic tickets fall below the threshold and go to human review.
- The KB retriever is dense-only (no BM25, no reranker) and the reply step uses a small local LLM.

## Suggested repository layout

```
your-project/
├── README.md            # from TEMPLATE_README.md
├── pyproject.toml       # pinned dependencies (uv lock)
├── data/                # or a script that downloads it; never commit personal data
├── src/pipeline.py      # assist(text) -> dict: the single entry point
├── src/train.py         # fine-tuning, with a config file for hyperparameters
├── eval/                # predictions CSVs + evaluate.py output
├── app.py               # demo
└── notebooks/           # exploration (optional)
```

## Timeline

| When | Do |
|---|---|
| End of Day 3 (10') | Confirm scope and metric with the instructor |
| Block 1 (1–2 h) | Data, splits, eval set, baseline number (reuse Lab 1.4 / 2.4 code) |
| Block 2 (1–2 h) | Improvement + fine-tuning (Lab 3.2 / 3.3 code) + leaderboard (Lab 3.4) |
| Block 3 (1 h) | Demo, README, 6–8 slides, one timed rehearsal, recorded backup of the demo |

Submit the repository link before Day 4, Session 1.
