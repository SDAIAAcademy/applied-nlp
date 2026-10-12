"""Prepare the course datasets.

Downloads the Bitext customer-support dataset (English, 27 intents, 11 categories)
from the Hugging Face Hub, cleans it, adds a business-rule `priority` label, and
writes stratified splits to ./data. Also post-processes the hand-written Arabic
test set (data/arabic_tickets.csv).

    uv run python prepare_data.py            # or: python prepare_data.py

Outputs (all in ./data):
    tickets_train.csv     200 per intent  (5,400)  - training classifiers
    tickets_val.csv        40 per intent  (1,080)  - model selection, thresholds
    tickets_test.csv       40 per intent  (1,080)  - final evaluation only
    tickets_history.csv    8,000 resolved tickets with agent responses - search / RAG
    tickets_ar_test.csv    108 Arabic tickets (MSA + Gulf) with the same labels
    labels.json            label lists and the intent -> category map
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

SEED = 42
DATA = Path(__file__).parent / "data"
DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

N_TRAIN, N_VAL, N_TEST, N_HISTORY = 200, 40, 40, 8000

# ---------------------------------------------------------------------------
# Priority: a business rule, not a linguistic label.
# Base level per intent, bumped one level when the text shows urgency or anger.
# ---------------------------------------------------------------------------
PRIORITIES = ["low", "medium", "high", "urgent"]
BASE_PRIORITY = {
    "payment_issue": "high",
    "complaint": "high",
    "get_refund": "medium",
    "track_refund": "medium",
    "cancel_order": "medium",
    "change_order": "medium",
    "change_shipping_address": "medium",
    "track_order": "medium",
    "recover_password": "medium",
    "registration_problems": "medium",
    "delete_account": "medium",
    "contact_human_agent": "medium",
}  # everything else: low
URGENCY_EN = re.compile(
    r"\b(urgent|urgently|asap|immediately|right now|now!|at once|emergency|furious|"
    r"unacceptable|ridiculous|worst|scam|fraud|lawyer|sue)\b|!!",
    re.IGNORECASE,
)
URGENCY_AR = re.compile(r"(عاجل|فورا|فوراً|حالا|حالاً|بسرعة|الحين|!!|شي ما يصير|سيئة)")


def priority_for(intent: str, text: str, offensive: bool = False) -> str:
    level = PRIORITIES.index(BASE_PRIORITY.get(intent, "low"))
    if offensive or URGENCY_EN.search(text) or URGENCY_AR.search(text):
        level = min(level + 1, len(PRIORITIES) - 1)
    return PRIORITIES[level]


# ---------------------------------------------------------------------------
# Placeholders: Bitext uses {{Order Number}} etc. We fill the ones in customer
# messages with realistic random values (so extraction exercises are meaningful)
# and the ones in agent responses with fixed DemoMart values.
# ---------------------------------------------------------------------------
CITIES = ["Riyadh", "Jeddah", "Dammam", "Makkah", "Madinah", "Abha", "Tabuk", "Khobar"]
COUNTRIES = ["Saudi Arabia", "UAE", "Kuwait", "Bahrain", "Oman", "Qatar"]
NAMES = ["Sara", "Fahad", "Noura", "Abdullah", "Reem", "Khalid", "Lama", "Omar", "Hessa", "Yousef"]
ACCOUNT_TYPES = ["Premium", "Business", "Gold", "Standard", "Platinum", "Basic"]


def fill_message_placeholders(text: str, rng: random.Random, values: dict) -> str:
    generators = {
        "order number": lambda: f"A-{rng.randint(10000, 99999)}",
        "invoice number": lambda: f"INV-2024-{rng.randint(1, 99999):05d}",
        "account type": lambda: rng.choice(ACCOUNT_TYPES),
        "account category": lambda: rng.choice(ACCOUNT_TYPES),
        "person name": lambda: rng.choice(NAMES),
        "delivery city": lambda: rng.choice(CITIES),
        "delivery country": lambda: rng.choice(COUNTRIES),
        "refund amount": lambda: str(rng.choice([49, 120, 199, 250, 349, 799, 1250])),
        "currency symbol": lambda: "SAR",
    }

    def repl(m: re.Match) -> str:
        key = m.group(1).strip().lower()
        if key not in values and key in generators:
            values[key] = generators[key]()
        return values.get(key, m.group(1))

    return re.sub(r"\{\{([^}]+)\}\}", repl, text)


RESPONSE_VALUES = {
    "online order interaction": "order management page",
    "customer support phone number": "920000000",
    "website url": "demomart.example",
    "customer support hours": "8:00 to 22:00, Saturday to Thursday",
    "online company portal info": "DemoMart account",
    "tracking number": "the tracking number in your confirmation SMS",
    "date range": "the last 30 days",
    "client last name": "",
    "salutation": "",
    "settings": "Settings",
    "profile": "Profile",
    "account change": "account change",
    "upgrade account": "Upgrade account",
    "login page url": "demomart.example/login",
    "store location": "the nearest pickup point",
    "forgot password": "Forgot password",
    "order status": "Order status",
    "customer support email": "support@demomart.example",
    "profile type": "account type",
    "order tracking": "Order tracking",
    "client name": "",
    "client first name": "",
    "client full name": "",
    "account recovery page url": "demomart.example/recover",
    "money amount": "the amount",
    "company name": "DemoMart",
    "company": "DemoMart",
    "live chat support": "live chat",
}


def fill_response_placeholders(text: str, values: dict) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1).strip().lower()
        if key in values:
            return values[key]
        if key in RESPONSE_VALUES:
            return RESPONSE_VALUES[key]
        return m.group(1).strip()  # fallback: keep the words, drop the braces

    out = re.sub(r"\{\{([^}]+)\}\}", repl, text)
    return re.sub(r"[ \t]{2,}", " ", out).replace(" ,", ",")


def stratified_take(df: pd.DataFrame, n_per_class: int, rng_seed: int) -> pd.DataFrame:
    # every intent has ~900 rows, far more than any split needs
    return df.groupby("intent").sample(n=n_per_class, random_state=rng_seed).reset_index(drop=True)


def main() -> None:
    from datasets import load_dataset

    DATA.mkdir(exist_ok=True)
    print(f"Downloading {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID, split="train").to_pandas()
    raw = raw.rename(columns={"instruction": "text"})

    # Normalise whitespace and drop exact duplicates (case-insensitive)
    raw["text"] = raw["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    raw = raw[raw["text"].str.len() > 0]
    raw = raw.loc[~raw["text"].str.lower().duplicated()].reset_index(drop=True)

    rng = random.Random(SEED)
    texts, responses = [], []
    for t, r in zip(raw["text"], raw["response"]):
        values: dict = {}
        texts.append(fill_message_placeholders(t, rng, values))
        responses.append(fill_response_placeholders(r, values))
    raw["text"], raw["response"] = texts, responses

    raw["offensive"] = raw["flags"].str.contains("W")
    raw["priority"] = [priority_for(i, t, o) for i, t, o in zip(raw["intent"], raw["text"], raw["offensive"])]
    raw["lang"] = "en"
    raw["ticket_id"] = range(1, len(raw) + 1)
    raw = raw[["ticket_id", "text", "intent", "category", "priority", "lang", "flags", "response"]]

    # Stratified, disjoint splits: test -> val -> train -> history
    test = stratified_take(raw, N_TEST, SEED)
    rest = raw[~raw.ticket_id.isin(test.ticket_id)]
    val = stratified_take(rest, N_VAL, SEED)
    rest = rest[~rest.ticket_id.isin(val.ticket_id)]
    train = stratified_take(rest, N_TRAIN, SEED)
    rest = rest[~rest.ticket_id.isin(train.ticket_id)]
    history = rest.sample(n=min(N_HISTORY, len(rest)), random_state=SEED).sort_values("ticket_id")

    cols = ["ticket_id", "text", "intent", "category", "priority", "lang"]
    train[cols].sample(frac=1, random_state=SEED).to_csv(DATA / "tickets_train.csv", index=False)
    val[cols].sample(frac=1, random_state=SEED).to_csv(DATA / "tickets_val.csv", index=False)
    test[cols].sample(frac=1, random_state=SEED).to_csv(DATA / "tickets_test.csv", index=False)
    history[cols + ["response"]].to_csv(DATA / "tickets_history.csv", index=False)

    # Arabic test set: add category + priority with the same rules
    intent2cat = raw.groupby("intent")["category"].first().to_dict()
    ar = pd.read_csv(DATA / "arabic_tickets.csv")
    unknown = set(ar["intent"]) - set(intent2cat)
    assert not unknown, f"Unknown intents in arabic_tickets.csv: {unknown}"
    ar.insert(0, "ticket_id", range(900001, 900001 + len(ar)))
    ar["category"] = ar["intent"].map(intent2cat)
    ar["priority"] = [priority_for(i, t) for i, t in zip(ar["intent"], ar["text"])]
    ar["lang"] = "ar"
    ar[cols + ["dialect"]].to_csv(DATA / "tickets_ar_test.csv", index=False)

    labels = {
        "intents": sorted(intent2cat),
        "categories": sorted(set(intent2cat.values())),
        "priorities": PRIORITIES,
        "intent_to_category": dict(sorted(intent2cat.items())),
    }
    (DATA / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"train {len(train)} | val {len(val)} | test {len(test)} | history {len(history)} | arabic {len(ar)}")
    print("priority distribution (train):", train["priority"].value_counts().to_dict())
    print(f"Wrote files to {DATA.resolve()}")


if __name__ == "__main__":
    main()
