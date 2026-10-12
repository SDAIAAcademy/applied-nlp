# <Project name>

**Team:** <names> · **Use case:** <1 / 2 / 3> · **Demo:** `uv run python app.py`

## Problem
<Who has this problem, why it matters, in 2–3 sentences.>

**Our system takes** <input> **and produces** <output>, **measured by** <metric> on <test set>.

## Data
| Split | Size | Languages / slices | Source |
|---|---|---|---|
| train | | | |
| validation | | | |
| test | | | |

Labels: <list and definitions or link>. Known limitations: <imbalance, noise, templated text, missing dialects...>.
How we prevented leakage: <near-duplicate check, grouping, ...>.

## Approach
<Architecture diagram or bullet pipeline.> Why these models: <quality, latency, cost, language coverage>.

## Results (held-out test set)
| System | Macro-F1 (all) | Macro-F1 (AR) | p95 latency | Cost / 1k | Notes |
|---|---|---|---|---|---|
| Baseline: <...> | | | | | |
| <Improvement> | | | | | |
| <Fine-tuned / adapted> | | | | | |

Confidence intervals: <bootstrap 95% CI from evaluate.py>. Fine-tuning: <before → after, or why we didn't fine-tune>.

## Error analysis
| Cause | Count (of N reviewed) | Example | What we'd do |
|---|---|---|---|
| Label noise | | | |
| Ambiguous | | | |
| Missing coverage (dialect, typos...) | | | |
| Model error | | | |
| Out of scope | | | |

## Reproduce
```bash
uv sync
uv run python src/train.py --config configs/final.yaml
uv run python eval/evaluate.py eval/predictions.csv --slice lang
uv run python app.py
```
Base model revisions: <model@commit>. Seeds: <...>.

## Limitations and next steps
<What would break first with real customer data? What would you do with two more weeks?>
