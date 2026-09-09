import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/golden/golden_set.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "true_intent" not in df.columns:
        raise ValueError("Golden set needs a true_intent column.")

    majority = df["true_intent"].value_counts().idxmax()
    pred = [majority] * len(df)

    p, r, f1, _ = precision_recall_fscore_support(
        df["true_intent"], pred, average="macro", zero_division=0
    )

    print("Majority intent:", majority)
    print("Accuracy:", accuracy_score(df["true_intent"], pred))
    print("Macro Precision:", p)
    print("Macro Recall:", r)
    print("Macro F1:", f1)

if __name__ == "__main__":
    main()
