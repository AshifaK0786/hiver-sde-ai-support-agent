import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/verizon_candidate_labels.csv")
OUTPUT = Path("data/golden/verizon_golden_200.csv")

TARGETS = {
    "general_support": 30,
    "internet_connectivity": 30,
    "billing_payment": 25,
    "service_outage": 25,
    "account_authentication": 20,
    "order_equipment": 20,
    "plan_upgrade": 20,
    "unknown": 30,
}

df = pd.read_csv(INPUT)

samples = []

for intent, n in TARGETS.items():
    subset = df[df["candidate_intent"] == intent]

    if len(subset) == 0:
        print(f"WARNING: No examples found for {intent}")
        continue

    sample_n = min(n, len(subset))

    sampled = subset.sample(
        n=sample_n,
        random_state=42
    ).copy()

    samples.append(sampled)

golden = pd.concat(samples, ignore_index=True)

golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Columns that we will fill during manual review
golden["gold_intent"] = ""
golden["expected_action"] = ""
golden["label_notes"] = ""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

golden.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GOLDEN SET CREATED")
print("=" * 60)

print(f"Total examples: {len(golden)}")
print(f"Saved to: {OUTPUT}")

print("\nCandidate distribution:")
print(golden["candidate_intent"].value_counts())

print("\nColumns:")
print(list(golden.columns))