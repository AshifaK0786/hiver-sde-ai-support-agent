import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


INPUT = Path("data/processed/verizon_train.csv")
OUTPUT = Path("models/intent_classifier.joblib")

df = pd.read_csv(INPUT)

X = df["customer_text"].astype(str)
y = df["intent"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=" * 60)
print("TRAINING DATA")
print("=" * 60)
print(f"Total examples : {len(df)}")
print(f"Training       : {len(X_train)}")
print(f"Test           : {len(X_test)}")

print("\nIntent distribution:")
print(y.value_counts())

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

print("\nTraining model...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")

print("\n" + "=" * 60)
print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, OUTPUT)

print("\nModel saved:")
print(OUTPUT)
