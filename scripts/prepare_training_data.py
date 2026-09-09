import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/verizon_candidate_labels.csv")
OUTPUT = Path("data/processed/verizon_train.csv")

df = pd.read_csv(INPUT)

# Remove uncertain examples
df = df[
    df["candidate_intent"].notna()
    & (df["candidate_intent"] != "unknown")
].copy()

# Keep only the columns needed for training
df = df[
    ["customer_text", "candidate_intent", "support_text"]
].rename(
    columns={"candidate_intent": "intent"}
)

# Remove duplicates and empty messages
df = df.dropna(subset=["customer_text", "intent"])
df = df.drop_duplicates(subset=["customer_text"])

# Limit dataset so training remains comfortably under 15 minutes
MAX_ROWS = 15000

if len(df) > MAX_ROWS:
    df = df.sample(
        MAX_ROWS,
        random_state=42
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("TRAINING DATA CREATED")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Saved: {OUTPUT}")

print("\nIntent distribution:")
print(df["intent"].value_counts())
