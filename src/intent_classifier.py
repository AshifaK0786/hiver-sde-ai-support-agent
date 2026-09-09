from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@dataclass
class Prediction:
    intent: str
    confidence: float

class TfidfIntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, texts: List[str], labels: List[str]):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        return self

    def predict_one(self, text: str) -> Prediction:
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        return Prediction(
            intent=str(self.model.classes_[idx]),
            confidence=float(probs[idx])
        )

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
