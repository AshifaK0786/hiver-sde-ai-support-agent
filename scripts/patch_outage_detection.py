from pathlib import Path

path = Path("src/support_agent.py")
text = path.read_text(encoding="utf-8")

old = '''        intent = self.intent_model.predict(
            [message]
        )[0]

        probabilities = self.intent_model.predict_proba(
            [message]
        )[0]

        confidence = max(probabilities)

        return intent, float(confidence)
'''

new = '''        intent = self.intent_model.predict(
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
'''

if old not in text:
    raise SystemExit(
        "Could not find the expected classify_intent block."
    )

path.write_text(
    text.replace(old, new),
    encoding="utf-8"
)

print("Updated outage detection successfully.")
