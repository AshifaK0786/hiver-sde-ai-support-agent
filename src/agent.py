import argparse
import json

from src.intent_classifier import TfidfIntentClassifier
from src.retrieval import HistoricalRetriever
from src.response_generator import generate_reply
from src.escalation import decide_escalation

class SupportAgent:
    def __init__(self, classifier, retriever):
        self.classifier = classifier
        self.retriever = retriever

    def run(self, message):
        prediction = self.classifier.predict_one(message)
        retrieved = self.retriever.search(message, k=5)

        top_score = retrieved[0].score if retrieved else 0.0
        has_resolution = bool(retrieved and retrieved[0].resolution)

        decision = decide_escalation(
            intent_confidence=prediction.confidence,
            retrieval_similarity=top_score,
            has_resolution=has_resolution
        )

        reply = generate_reply(message, retrieved)

        return {
            "message": message,
            "intent": prediction.intent,
            "confidence": prediction.confidence,
            "retrieved_cases": [
                {
                    "score": r.score,
                    "text": r.text,
                    "resolution": r.resolution
                }
                for r in retrieved
            ],
            "reply": reply,
            "decision": decision
        }

def demo():
    print("This CLI is a demonstration shell.")
    print("Train/load a classifier and retrieval index from your selected-brand data before using it for final evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="")
    args = parser.parse_args()
    if not args.message:
        raise SystemExit("Use --message 'your customer message'")
    demo()
