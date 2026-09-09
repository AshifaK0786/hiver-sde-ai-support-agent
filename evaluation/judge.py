import json

JUDGE_RUBRIC = {
    "correctness": "Does the reply correctly address the customer's problem?",
    "groundedness": "Is the reply supported by the supplied historical evidence?",
    "helpfulness": "Does it provide a useful next step?",
    "tone": "Is the tone professional and appropriate?",
    "hallucination": "Does it avoid unsupported policies, promises, or facts?"
}

def build_judge_prompt(customer, evidence, reply):
    return f"""Rate the following customer-support reply from 1 to 5 for each criterion.

Customer:
{customer}

Historical evidence:
{json.dumps(evidence, ensure_ascii=False)}

Reply:
{reply}

Criteria:
{json.dumps(JUDGE_RUBRIC, indent=2)}

Return JSON with numeric scores and a short justification for each criterion.
"""
