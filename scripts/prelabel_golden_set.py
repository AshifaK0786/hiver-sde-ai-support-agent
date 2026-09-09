import pandas as pd
import re
from pathlib import Path

INPUT = Path("data/golden/verizon_golden_200.csv")
OUTPUT = Path("data/golden/verizon_golden_200_review.csv")

def classify(text):
    text = str(text).lower()

    if any(x in text for x in [
        "outage", "down", "no service", "service is down",
        "everyone is down", "all of verizon is down"
    ]):
        return "service_outage"

    if any(x in text for x in [
        "wifi", "wi-fi", "internet", "router", "connection",
        "connect", "disconnect", "offline", "fios"
    ]):
        return "internet_connectivity"

    if any(x in text for x in [
        "bill", "billing", "charged", "charge", "payment",
        "pay my", "autopay", "refund", "refunds", "cost"
    ]):
        return "billing_payment"

    if any(x in text for x in [
        "login", "log in", "password", "authenticate",
        "authentication", "verify account", "verification",
        "validate account", "can't access account"
    ]):
        return "account_authentication"

    if any(x in text for x in [
        "order", "delivery", "delivered", "shipping",
        "package", "equipment", "phone arrived", "new phone"
    ]):
        return "order_equipment"

    if any(x in text for x in [
        "upgrade", "upgrading", "plan", "unlimited",
        "contract", "new plan", "change my plan"
    ]):
        return "plan_upgrade"

    return "general_support"


def action_for(intent, text):
    text = str(text).lower()

    sensitive = [
        "password", "account number", "social security",
        "ssn", "credit card", "bank account",
        "private", "personal information"
    ]

    if intent in [
        "billing_payment",
        "account_authentication",
        "order_equipment"
    ]:
        return "human_escalation"

    if any(x in text for x in sensitive):
        return "human_escalation"

    if intent == "service_outage":
        return "auto_handle"

    if intent == "internet_connectivity":
        return "auto_handle"

    if intent == "plan_upgrade":
        return "human_escalation"

    return "auto_handle"


df = pd.read_csv(INPUT)

df["suggested_gold_intent"] = df["customer_text"].apply(classify)

df["suggested_expected_action"] = df.apply(
    lambda row: action_for(
        row["suggested_gold_intent"],
        row["customer_text"]
    ),
    axis=1
)

df["human_verified"] = ""
df["human_notes"] = ""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GOLDEN SET PRELABELING COMPLETE")
print("=" * 60)
print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print()
print("Suggested intent distribution:")
print(df["suggested_gold_intent"].value_counts())
print()
print("IMPORTANT:")
print("These are AI/keyword suggestions, NOT final human labels.")
print("Review and correct the labels before using this as the golden set.")
