import json
from typing import List

from src.config import OPENAI_API_KEY, OPENAI_MODEL

def build_prompt(customer_message: str, cases: List[dict]) -> str:
    evidence = "\n\n".join(
        f"CASE {i+1}\nCustomer: {c['text']}\nHistorical resolution: {c['resolution']}"
        for i, c in enumerate(cases)
    )

    return f"""You are a customer-support assistant for one specific brand.

Answer the customer using the historical support evidence below.
Do not invent policies, prices, timelines, account actions, guarantees, or facts.
If the evidence is insufficient to safely answer, say that escalation is appropriate.

Historical evidence:
{evidence}

New customer message:
{customer_message}

Return JSON:
{{
  "reply": "concise support reply",
  "grounded": true,
  "reason": "why the response is supported"
}}
"""

def generate_reply(customer_message: str, retrieved):
    cases = [
        {"text": r.text, "resolution": r.resolution}
        for r in retrieved
    ]

    if not OPENAI_API_KEY or not OPENAI_MODEL:
        if cases and cases[0]["resolution"]:
            return {
                "reply": cases[0]["resolution"],
                "grounded": True,
                "reason": "Deterministic fallback using the highest-ranked historical resolution."
            }
        return {
            "reply": "We need a little more information to help with this request.",
            "grounded": False,
            "reason": "No usable historical resolution was available."
        }

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a careful customer-support response generator."},
            {"role": "user", "content": build_prompt(customer_message, cases)}
        ],
    )

    return json.loads(response.choices[0].message.content)
