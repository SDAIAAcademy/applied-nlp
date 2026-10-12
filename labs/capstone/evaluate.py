"""Evaluate a predictions file: overall and per-slice metrics with bootstrap CIs.

    uv run python capstone/evaluate.py predictions.csv [--slice lang] [--slice dialect]

The CSV needs columns `y_true` and `y_pred`; any other column can be used as a slice.
Copy the printed tables into your README.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nlp_utils as U  # noqa: E402


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def summarize(df: pd.DataFrame) -> dict:
    f1, lo, hi = U.bootstrap_ci(df.y_true.astype(str), df.y_pred.astype(str), macro_f1, n=500)
    return {"n": len(df), "accuracy": round(accuracy_score(df.y_true, df.y_pred), 3),
            "macro_f1": round(f1, 3), "macro_f1_95ci": f"[{lo:.3f}, {hi:.3f}]"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--slice", action="append", default=[], help="column to slice by (repeatable)")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    df["y_pred"] = df["y_pred"].fillna("(abstain)")
    rows = [{"slice": "all", **summarize(df)}]
    for col in args.slice:
        for value, g in df.groupby(col):
            rows.append({"slice": f"{col}={value}", **summarize(g)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Most frequent confusions:")
    print(U.top_confusions(df.y_true, df.y_pred, k=10).to_string(index=False))
    print()
    print(classification_report(df.y_true, df.y_pred, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
