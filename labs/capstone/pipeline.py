"""Capstone starter: a small, working ticket assistant to adapt and improve.

    from pipeline import assist
    assist("Where is my refund?")

It wires the Day 1-2 building blocks together:
    classify (e5 embeddings + logistic regression, abstains below `tau`)
    -> category + priority (business rules)
    -> retrieve similar resolved tickets and help-center sections
    -> optional grounded reply from the LLM (use_llm=True)

Replace any stage with your own (fine-tuned encoder from Lab 3.2, cascade from
Lab 2.4, LoRA model from Lab 3.3, ...) and keep the output contract stable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # labs/ for nlp_utils + prepare_data
import nlp_utils as U  # noqa: E402
from prepare_data import priority_for  # noqa: E402

ORDER_ID = re.compile(r"\bA-\d{5}\b")


class TicketAssistant:
    def __init__(self, tau: float = 0.5, use_llm: bool = False, n_history: int | None = None):
        from sklearn.linear_model import LogisticRegression

        self.tau, self.use_llm = tau, use_llm
        self.labels = U.load_labels()

        train = U.load_tickets("train")
        X = U.embed(train.text, prefix="query: ", cache_key="train")
        self.clf = LogisticRegression(max_iter=3000, C=20.0).fit(X, train.intent)  # weaker regularization -> sharper probabilities

        history = U.load_tickets("history")
        if n_history:
            history = history.head(n_history)
        self.history = history.reset_index(drop=True)
        self.H = U.embed(self.history.text, prefix="passage: ", cache_key=f"history{len(history)}")

        self.kb = U.load_kb()
        self.K = U.embed([f"{s['title']} - {s['section']}: {s['text']}" for s in self.kb], prefix="passage: ")

    def classify(self, text: str) -> tuple[str | None, float]:
        p = self.clf.predict_proba(U.embed([text], prefix="query: "))[0]
        i = int(p.argmax())
        return (self.clf.classes_[i] if p[i] >= self.tau else None), float(p[i])

    def retrieve(self, text: str, k_history: int = 3, k_kb: int = 3) -> tuple[list[dict], list[dict]]:
        q = U.embed([text], prefix="query: ")[0]
        h_idx = np.argsort(-(self.H @ q))[:k_history]
        k_idx = np.argsort(-(self.K @ q))[:k_kb]
        similar = [{"ticket_id": int(self.history.ticket_id[i]), "text": self.history.text[i],
                    "response": self.history.response[i][:300]} for i in h_idx]
        kb = [{"id": self.kb[i]["id"], "text": self.kb[i]["text"]} for i in k_idx]
        return similar, kb

    def draft_reply(self, text: str, kb: list[dict]) -> dict:
        context = "\n".join(f"[{n}] ({s['id']}) {s['text']}" for n, s in enumerate(kb, start=1))
        messages = [
            {"role": "system", "content": (
                "You draft replies for customer-support agents. Use ONLY the numbered context and cite it as [1], [2]. "
                "If the context does not answer the ticket, set needs_human to true. Reply in the customer's language. "
                'Return JSON only: {"reply": str, "citations": [int], "needs_human": bool}')},
            {"role": "user", "content": f"CONTEXT\n{context}\n\nTICKET\n<ticket>{text}</ticket>"},
        ]
        raw = U.get_llm().chat(messages, max_new_tokens=300)
        try:
            out = json.loads(re.search(r"\{.*\}", raw, flags=re.S).group(0))
            out["citations"] = [c for c in out.get("citations", []) if isinstance(c, int) and 1 <= c <= len(kb)]
            if not out["citations"]:
                out["needs_human"] = True
            return out
        except (AttributeError, json.JSONDecodeError):
            return {"reply": None, "citations": [], "needs_human": True, "raw": raw}

    def __call__(self, text: str) -> dict:
        intent, conf = self.classify(text)
        similar, kb = self.retrieve(text)
        out = {
            "intent": intent,
            "confidence": round(conf, 3),
            "route": "auto" if intent else "human_review",
            "category": self.labels["intent_to_category"].get(intent) if intent else None,
            "priority": priority_for(intent, text) if intent else "medium",
            "language": "ar" if re.search(r"[؀-ۿ]", text) else "en",
            "order_id": (m.group(0) if (m := ORDER_ID.search(text)) else None),
            "similar_tickets": [s["ticket_id"] for s in similar],
            "kb_sections": [s["id"] for s in kb],
        }
        if self.use_llm:
            out.update(self.draft_reply(text, kb))
        return out


_ASSISTANT: TicketAssistant | None = None


def assist(text: str, **kwargs) -> dict:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = TicketAssistant(**kwargs)
    return _ASSISTANT(text)


if __name__ == "__main__":
    for t in ["I was charged twice for order A-55821, fix it now!", "وين طلبي؟ صار له أسبوع"]:
        print(json.dumps(assist(t, n_history=U.pick(None, 500)), ensure_ascii=False, indent=2))
