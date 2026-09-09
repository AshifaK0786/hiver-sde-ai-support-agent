from src.config import AUTO_INTENT_THRESHOLD, AUTO_SIMILARITY_THRESHOLD

def decide_escalation(
    intent_confidence: float,
    retrieval_similarity: float,
    has_resolution: bool,
    sensitive: bool = False,
    ambiguous: bool = False,
):
    if sensitive:
        return {
            "decision": "ESCALATE",
            "reason": "Sensitive/account-specific issue requires human review."
        }

    if ambiguous:
        return {
            "decision": "ESCALATE",
            "reason": "The message appears ambiguous or may contain multiple intents."
        }

    if intent_confidence < AUTO_INTENT_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "reason": "Intent confidence is below the auto-handling threshold."
        }

    if retrieval_similarity < AUTO_SIMILARITY_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "reason": "Historical-case similarity is too low for a grounded response."
        }

    if not has_resolution:
        return {
            "decision": "ESCALATE",
            "reason": "No reliable historical resolution was found."
        }

    return {
        "decision": "AUTO-HANDLE",
        "reason": "High-confidence intent and sufficiently similar historical resolution."
    }
