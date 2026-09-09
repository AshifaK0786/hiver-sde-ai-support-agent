import argparse
from pathlib import Path
import pandas as pd

CANDIDATE_TEXT = ["text", "tweet_text", "content", "message"]
CANDIDATE_BRAND = ["in_response_to_user_id", "brand", "company", "handle", "author"]
CANDIDATE_ID = ["tweet_id", "id"]
CANDIDATE_REPLY = ["in_response_to_status_id", "in_reply_to_status_id", "reply_to_id"]

def find_csvs(path):
    return sorted(Path(path).glob("*.csv"))

def detect_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None

def inspect(path):
    csvs = find_csvs(path)
    if not csvs:
        print(f"No CSV files found in {path}")
        return

    for csv in csvs:
        df = pd.read_csv(csv, nrows=5)
        print(f"\nFILE: {csv}")
        print("COLUMNS:", list(df.columns))

        text_col = detect_column(df.columns, CANDIDATE_TEXT)
        print("Detected text column:", text_col)

        full = pd.read_csv(csv)
        print("Rows:", len(full))
        print("Missing values:")
        print(full.isna().sum().sort_values(ascending=False).head(10))

        print("\nPotential categorical columns:")
        for col in full.columns:
            if full[col].dtype == "object":
                n = full[col].nunique(dropna=True)
                if n < 200:
                    print(f"  {col}: {n} unique")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    args = parser.parse_args()
    inspect(args.input)

if __name__ == "__main__":
    main()
