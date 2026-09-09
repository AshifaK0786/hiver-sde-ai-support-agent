import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/golden/golden_set.csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    required = {"message", "true_intent"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError("Train/test require message and true_intent columns.")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(train["message"], train["true_intent"])
    pred = pipe.predict(test["message"])

    print(classification_report(test["true_intent"], pred, zero_division=0))

if __name__ == "__main__":
    main()
