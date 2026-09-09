import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = Path("models")


class SupportAgent:

    def __init__(self):

        self.intent_model = joblib.load(
            MODEL_DIR / "intent_classifier.joblib"
        )

        self.vectorizer = joblib.load(
            MODEL_DIR / "retrieval_vectorizer.joblib"
        )

        self.matrix = joblib.load(
            MODEL_DIR / "retrieval_matrix.joblib"
        )

        self.history = joblib.load(
            MODEL_DIR / "retrieval_data.joblib"
        )

    # ---------------------------------------------------------
    # INTENT CLASSIFICATION
    # ---------------------------------------------------------

    def classify_intent(self, message):

        intent = self.intent_model.predict(
            [message]
        )[0]

        probabilities = self.intent_model.predict_proba(
            [message]
        )[0]

        confidence = max(probabilities)

        # Rule-based correction for strong outage signals.
        # This handles cases where words such as "internet"
        # dominate the ML classifier even though the customer
        # is describing a broader outage.

        text = message.lower()

        outage_signals = [
            "is verizon down",
            "verizon is down",
            "everything is down",
            "nothing is working",
            "all services",
            "service outage",
            "outage",
            "all of verizon is down"
        ]

        if any(signal in text for signal in outage_signals):
            intent = "service_outage"

        return intent, float(confidence)

    # ---------------------------------------------------------
    # HISTORICAL RETRIEVAL
    # ---------------------------------------------------------

    def retrieve_examples(
        self,
        message,
        intent,
        top_k=3
    ):

        # First filter historical examples
        # to the predicted intent.
        candidates = self.history[
            self.history["intent"] == intent
        ].copy()

        if len(candidates) == 0:

            candidates = self.history.copy()

        candidate_indices = candidates.index.tolist()

        candidate_matrix = self.matrix[
            candidate_indices
        ]

        query_vector = self.vectorizer.transform(
            [message]
        )

        similarities = cosine_similarity(
            query_vector,
            candidate_matrix
        )[0]

        top_positions = similarities.argsort()[
            -top_k:
        ][::-1]

        results = []

        for position in top_positions:

            original_index = candidate_indices[
                position
            ]

            row = self.history.iloc[
                original_index
            ]

            results.append({
                "customer_message":
                    row["customer_text"],

                "historical_response":
                    row["support_text"],

                "intent":
                    row["intent"],

                "similarity":
                    float(similarities[position])
            })

        return results

    # ---------------------------------------------------------
    # ESCALATION
    # ---------------------------------------------------------

    def decide_escalation(
        self,
        intent,
        confidence,
        retrieval_results
    ):

        sensitive_intents = {
            "billing_payment",
            "account_authentication",
            "order_equipment",
            "plan_upgrade"
        }

        if confidence < 0.60:

            return (
                True,
                "Low intent confidence"
            )

        if intent in sensitive_intents:

            return (
                True,
                "Issue may require account-specific "
                "or transactional handling"
            )

        if not retrieval_results:

            return (
                True,
                "No historical evidence found"
            )

        best_similarity = (
            retrieval_results[0]["similarity"]
        )

        if best_similarity < 0.20:

            return (
                True,
                "Historical examples are not "
                "sufficiently similar"
            )

        return (
            False,
            "High-confidence issue with relevant history"
        )

    # ---------------------------------------------------------
    # RESPONSE GENERATION
    # ---------------------------------------------------------

    def draft_response(
        self,
        message,
        intent,
        retrieval_results,
        escalate
    ):
        if escalate:
            if intent == "service_outage":
                return (
                    "Thanks for reaching out. We understand "
                    "you may be experiencing a service outage. "
                    "Please send us a private message so our "
                    "support team can check the service status "
                    "for your area and assist you."
                )

            return (
                "Thanks for reaching out. "
                "This issue may require account-specific "
                "assistance. Please send us a private "
                "message so our support team can securely "
                "look into this."
            )

        if not retrieval_results:
            return (
                "Thanks for reaching out. "
                "Could you provide a few more details "
                "about the issue so we can help?"
            )

        # Use the strongest historical response
        # as the grounded response baseline.
        return retrieval_results[0]["historical_response"]

    # ---------------------------------------------------------
    # COMPLETE AGENT
    # ---------------------------------------------------------

    def run(self, message):

        intent, confidence = self.classify_intent(
            message
        )

        examples = self.retrieve_examples(
            message,
            intent,
            top_k=3
        )

        escalate, reason = (
            self.decide_escalation(
                intent,
                confidence,
                examples
            )
        )

        response = self.draft_response(
            message,
            intent,
            examples,
            escalate
        )

        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "escalate": escalate,
            "escalation_reason": reason,
            "historical_examples": examples
        }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    agent = SupportAgent()

    print("=" * 70)
    print("VERIZON AI SUPPORT AGENT")
    print("=" * 70)

    while True:

        message = input(
            "\nCustomer message (type 'exit' to quit): "
        )

        if message.lower() == "exit":
            break

        result = agent.run(message)

        print("\nIntent:")
        print(result["intent"])

        print("\nConfidence:")
        print(
            f"{result['confidence']:.3f}"
        )

        print("\nDraft response:")
        print(result["response"])

        print("\nEscalate:")
        print(result["escalate"])

        print("\nReason:")
        print(result["escalation_reason"])

        print("\nHistorical examples:")

        for i, example in enumerate(
            result["historical_examples"],
            start=1
        ):

            print(
                f"\n{i}. Similarity: "
                f"{example['similarity']:.3f}"
            )

            print(
                "Customer:",
                example["customer_message"][:200]
            )

            print(
                "Support:",
                example["historical_response"][:200]
            )