import pandas as pd
import re
from pathlib import Path

INPUT = Path("data/processed/verizon_support_pairs.csv")
OUTPUT = Path("data/processed/verizon_candidate_labels.csv")

df = pd.read_csv(INPUT)

print(f"Loaded {len(df):,} Verizon conversations")


def contains_any(text, keywords):
    text = str(text).lower()
    return any(k in text for k in keywords)


def classify(text):

    text = str(text).lower()

    # --------------------------------------------------
    # 1. Service outage
    # --------------------------------------------------
    if contains_any(text, [
        "outage",
        "all of verizon is down",
        "service is down",
        "services are down",
        "everything is down",
        "network is down"
    ]):
        return "service_outage"

    # --------------------------------------------------
    # 2. Internet / connectivity
    # --------------------------------------------------
    if contains_any(text, [
        "internet",
        "wifi",
        "wi-fi",
        "router",
        "fios",
        "offline",
        "disconnect",
        "disconnected",
        "connection",
        "keeps going down",
        "keep going down",
        "cuts out"
    ]):
        return "internet_connectivity"

    # --------------------------------------------------
    # 3. Billing / payment
    # --------------------------------------------------
    if contains_any(text, [
        "bill",
        "billing",
        "payment",
        "pay my",
        "paying",
        "charged",
        "charge",
        "reimbursement",
        "refund",
        "autopay",
        "prorate",
        "pro-rate"
    ]):
        return "billing_payment"

    # --------------------------------------------------
    # 4. Account / authentication
    # --------------------------------------------------
    if contains_any(text, [
        "account",
        "authenticate",
        "authentication",
        "validate the account",
        "verify my",
        "verification",
        "can't log",
        "cannot log",
        "password",
        "username"
    ]):
        return "account_authentication"

    # --------------------------------------------------
    # 5. Order / equipment
    # --------------------------------------------------
    if contains_any(text, [
        "order",
        "delivery",
        "delivered",
        "shipping",
        "package",
        "equipment",
        "order number"
    ]):
        return "order_equipment"

    # --------------------------------------------------
    # 6. Plan / upgrade
    # --------------------------------------------------
    if contains_any(text, [
        "upgrade",
        "plan",
        "contract",
        "renew",
        "package",
        "unlimited data"
    ]):
        return "plan_upgrade"

    # --------------------------------------------------
    # 7. General support
    # --------------------------------------------------
    if contains_any(text, [
        "help",
        "support",
        "customer service",
        "agent",
        "representative",
        "assist"
    ]):
        return "general_support"

    return "unknown"


df["candidate_intent"] = df["customer_text"].apply(classify)

print("\n========== CANDIDATE INTENT DISTRIBUTION ==========")

counts = df["candidate_intent"].value_counts()

print(counts)

print("\n========== PERCENTAGE ==========")

percentages = (
    df["candidate_intent"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(percentages)

# Save
df.to_csv(OUTPUT, index=False)

print(f"\nSaved to: {OUTPUT}")

# --------------------------------------------------
# Show examples
# --------------------------------------------------

print("\n========== EXAMPLES ==========")

for intent in counts.index:

    print(f"\n### {intent}")

    examples = df[
        df["candidate_intent"] == intent
    ].head(3)

    for _, row in examples.iterrows():

        print("\nCustomer:")
        print(row["customer_text"])

        print("Support:")
        print(row["support_text"])

print("\nCompleted.")