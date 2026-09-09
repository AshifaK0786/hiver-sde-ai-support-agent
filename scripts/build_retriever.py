import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA = Path("data/processed/verizon_train.csv")
MODEL_DIR = Path("models")

VECTORIZER_PATH = MODEL_DIR / "retrieval_vectorizer.joblib"
MATRIX_PATH = MODEL_DIR / "retrieval_matrix.joblib"
DATA_PATH = MODEL_DIR / "retrieval_data.joblib"


df = pd.read_csv(DATA)

df = df.dropna(
    subset=["customer_text", "support_text"]
).copy()

df["customer_text"] = df["customer_text"].astype(str)
df["support_text"] = df["support_text"].astype(str)

print("=" * 60)
print("BUILDING HISTORICAL RESPONSE RETRIEVER")
print("=" * 60)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

matrix = vectorizer.fit_transform(df["customer_text"])

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(matrix, MATRIX_PATH)

joblib.dump(
    df[
        [
            "customer_text",
            "support_text",
            "intent"
        ]
    ],
    DATA_PATH
)

print(f"Examples indexed : {len(df)}")
print(f"Vectorizer       : {VECTORIZER_PATH}")
print(f"Matrix           : {MATRIX_PATH}")
print(f"Historical data  : {DATA_PATH}")

print("\nRetriever ready.")
