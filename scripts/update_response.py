from pathlib import Path

path = Path("src/support_agent.py")

text = path.read_text(encoding="utf-8")

start = text.index("    def draft_response(")
end = text.index("    # ---------------------------------------------------------", start + 10)

new_method = '''    def draft_response(
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

'''

path.write_text(
    text[:start] + new_method + text[end:],
    encoding="utf-8"
)

print("draft_response() updated successfully.")
