import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/twcs/twcs.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BRAND = "VerizonSupport"


def normalize_id(value):
    """Convert IDs such as 3.0 -> 3 while preserving missing values."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


print("Loading dataset...")

df = pd.read_csv(INPUT)

print(f"Total rows: {len(df):,}")

# ---------------------------------------------------------
# Normalize important columns
# ---------------------------------------------------------

df["tweet_id_norm"] = df["tweet_id"].apply(normalize_id)
df["response_tweet_id_norm"] = df["response_tweet_id"].apply(normalize_id)
df["in_response_to_tweet_id_norm"] = (
    df["in_response_to_tweet_id"].apply(normalize_id)
)

# Normalize inbound
df["inbound_norm"] = (
    df["inbound"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ---------------------------------------------------------
# Select VerizonSupport tweets
# ---------------------------------------------------------

brand_df = df[
    df["author_id"].astype(str).str.strip() == BRAND
].copy()

print(f"\n{BRAND} tweets: {len(brand_df):,}")

# ---------------------------------------------------------
# Create tweet lookup
# ---------------------------------------------------------

tweet_lookup = (
    df.drop_duplicates("tweet_id_norm")
    .set_index("tweet_id_norm")
)

records = []

# ---------------------------------------------------------
# Match Verizon replies to customer messages
# ---------------------------------------------------------

for _, support in brand_df.iterrows():

    parent_id = support["in_response_to_tweet_id_norm"]

    if parent_id is None:
        continue

    if parent_id not in tweet_lookup.index:
        continue

    customer = tweet_lookup.loc[parent_id]

    # Customer tweet should be inbound
    if customer["inbound_norm"] != "true":
        continue

    records.append({
        "brand": BRAND,

        "customer_tweet_id": parent_id,
        "customer_author_id": customer["author_id"],
        "customer_text": customer["text"],
        "customer_created_at": customer["created_at"],

        "support_tweet_id": support["tweet_id_norm"],
        "support_text": support["text"],
        "support_created_at": support["created_at"],
    })


result = pd.DataFrame(records)

output = OUTPUT_DIR / "verizon_support_pairs.csv"

result.to_csv(output, index=False)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n========== RESULTS ==========")

print(f"Customer-support pairs: {len(result):,}")

print(f"Output: {output}")

if len(result) > 0:

    print("\n========== FIRST 10 EXAMPLES ==========")

    for _, row in result.head(10).iterrows():

        print("\n----------------------------------------")

        print("CUSTOMER:")
        print(row["customer_text"])

        print("\nVERIZON SUPPORT:")
        print(row["support_text"])

print("\nExtraction completed.")