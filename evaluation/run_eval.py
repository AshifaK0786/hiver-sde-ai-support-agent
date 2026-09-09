import argparse
import pandas as pd
from sklearn.metrics import classification_report
from baselines.tfidf import Pipeline, TfidfVectorizer, LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--golden", default="data/golden/golden_set.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    golden = pd.read_csv(args.golden)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(train["message"], train["true_intent"])
    pred = pipe.predict(golden["message"])

    print("=== TF-IDF baseline on golden set ===")
    print(classification_report(golden["true_intent"], pred, zero_division=0))

if __name__ == "__main__":
    main()
