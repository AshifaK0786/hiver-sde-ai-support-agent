from src.escalation import decide_escalation

def test_low_similarity_escalates():
    result = decide_escalation(
        intent_confidence=0.95,
        retrieval_similarity=0.20,
        has_resolution=True
    )
    assert result["decision"] == "ESCALATE"

def test_high_confidence_can_auto_handle():
    result = decide_escalation(
        intent_confidence=0.95,
        retrieval_similarity=0.80,
        has_resolution=True
    )
    assert result["decision"] == "AUTO-HANDLE"

def test_sensitive_escalates():
    result = decide_escalation(
        intent_confidence=0.99,
        retrieval_similarity=0.99,
        has_resolution=True,
        sensitive=True
    )
    assert result["decision"] == "ESCALATE"
