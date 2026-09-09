import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/raw/twcs/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

print("\n========== DATASET INFO ==========")
print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\n========== INBOUND DISTRIBUTION ==========")
print(df["inbound"].value_counts(dropna=False))

df["inbound_bool"] = (
    df["inbound"]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })
)

print("\n========== AUTHOR ANALYSIS ==========")

company_tweets = df[df["inbound_bool"] == False].copy()

print("Possible company/support tweets:", len(company_tweets))

print("\nTop author IDs among possible company replies:")

author_counts = (
    company_tweets["author_id"]
    .value_counts()
    .head(30)
)

print(author_counts.to_string())

print("\n========== SAMPLE COMPANY TWEETS ==========")

sample = company_tweets[
    ["author_id", "text", "in_response_to_tweet_id"]
].head(30)

for _, row in sample.iterrows():
    print("\n--------------------------------")
    print("Author:", row["author_id"])
    print("Text:", row["text"])
    print("Reply to:", row["in_response_to_tweet_id"])

print("\nAnalysis completed.")