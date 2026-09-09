import argparse
import json
from pathlib import Path
import pandas as pd

from src.preprocessing import normalize_dataframe

TEXT_CANDIDATES = ["text", "tweet_text", "content", "message"]
ID_CANDIDATES = ["tweet_id", "id"]
REPLY_CANDIDATES = ["in_response_to_status_id", "in_reply_to_status_id", "reply_to_id"]
BRAND_CANDIDATES = ["brand", "company", "handle"]

def detect(columns, candidates):
    low = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def build(input_dir, output):
    files = sorted(Path(input_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {input_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        text_col = detect(df.columns, TEXT_CANDIDATES)
        if not text_col:
            continue
        df = normalize_dataframe(df, text_col)
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        raise ValueError("Could not detect a usable text column.")

    df = pd.concat(frames, ignore_index=True)

    id_col = detect(df.columns, ID_CANDIDATES)
    reply_col = detect(df.columns, REPLY_CANDIDATES)
    brand_col = detect(df.columns, BRAND_CANDIDATES)

    # Conservative reconstruction: use reply-to IDs when available.
    # If the dataset lacks the required fields, each row remains its own case
    # rather than inventing conversation relationships.
    records = []

    for i, row in df.iterrows():
        tweet_id = str(row[id_col]) if id_col else str(i)
        parent = str(row[reply_col]) if reply_col and pd.notna(row[reply_col]) else None
        brand = str(row[brand_col]) if brand_col and pd.notna(row[brand_col]) else None

        records.append({
            "message_id": tweet_id,
            "parent_id": parent,
            "brand": brand,
            "text": row["clean_text"],
            "source_file": row["source_file"],
        })

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} normalized messages to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed/messages.jsonl")
    args = parser.parse_args()
    build(args.input, args.output)
