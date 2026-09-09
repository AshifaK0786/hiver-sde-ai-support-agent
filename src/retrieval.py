from dataclasses import dataclass
from typing import List, Dict
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer

@dataclass
class RetrievedCase:
    score: float
    text: str
    resolution: str
    metadata: Dict

class HistoricalRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.cases = []

    def fit(self, cases: List[Dict]):
        self.cases = cases
        texts = [
            f"{c.get('customer_text','')} {c.get('resolution','')}"
            for c in cases
        ]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        if faiss is not None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index = embeddings
        return self

    def search(self, query: str, k=5):
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        if faiss is not None and hasattr(self.index, "search"):
            scores, ids = self.index.search(q, min(k, len(self.cases)))
            pairs = zip(scores[0], ids[0])
        else:
            scores = self.index @ q[0]
            ids = np.argsort(-scores)[:k]
            pairs = ((scores[i], i) for i in ids)

        results = []
        for score, idx in pairs:
            if idx < 0:
                continue
            case = self.cases[int(idx)]
            results.append(
                RetrievedCase(
                    score=float(score),
                    text=case.get("customer_text", ""),
                    resolution=case.get("resolution", ""),
                    metadata=case
                )
            )
        return results
