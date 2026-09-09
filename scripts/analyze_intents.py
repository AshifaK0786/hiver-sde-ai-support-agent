import pandas as pd
from pathlib import Path
from collections import Counter
import re

INPUT = Path("data/processed/verizon_support_pairs.csv")

print("Loading Verizon support pairs...")

df = pd.read_csv(INPUT)

print(f"Rows: {len(df):,}")

# ---------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------

print("\n========== BASIC STATISTICS ==========")

print("Unique customers:", df["customer_author_id"].nunique())
print("Unique support responses:", df["support_text"].nunique())

# ---------------------------------------------------------
# Common words
# ---------------------------------------------------------

print("\n========== COMMON WORDS ==========")

texts = df["customer_text"].fillna("").astype(str).str.lower()

stopwords = {
    "the", "and", "to", "a", "of", "is", "i", "it",
    "for", "in", "on", "my", "me", "you", "this",
    "that", "with", "have", "can", "be", "was",
    "are", "at", "but", "or", "we", "your", "please",
    "do", "just", "so", "im", "I'm".lower()
}

counter = Counter()

for text in texts:

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)

    for word in words:

        if word not in stopwords:
            counter[word] += 1

print("\nTop 100 words:")

for word, count in counter.most_common(100):
    print(f"{word:25s} {count}")

# ---------------------------------------------------------
# Keyword category analysis
# ---------------------------------------------------------

categories = {
    "internet_wifi": [
        "internet", "wifi", "wi-fi", "router",
        "connection", "connect", "fios"
    ],

    "billing_payment": [
        "bill", "billing", "payment", "pay",
        "charge", "charged", "price", "cost"
    ],

    "account_login": [
        "account", "login", "log in", "password",
        "username", "authenticate", "verification"
    ],

    "order_delivery": [
        "order", "delivery", "delivered", "shipping",
        "package", "equipment"
    ],

    "mobile_phone": [
        "phone", "iphone", "android", "mobile",
        "cell", "device", "sim"
    ],

    "service_outage": [
        "outage", "out", "down", "offline",
        "disconnect", "disconnected", "not working"
    ],

    "upgrade_plan": [
        "upgrade", "plan", "contract", "renew",
        "new plan", "change plan"
    ],

    "customer_service": [
        "customer service", "support", "agent",
        "representative", "help"
    ]
}

print("\n========== KEYWORD CATEGORY ANALYSIS ==========")

category_counts = {}

for category, keywords in categories.items():

    count = 0

    for text in texts:

        if any(keyword in text for keyword in keywords):
            count += 1

    category_counts[category] = count

for category, count in sorted(
    category_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = count / len(df) * 100

    print(
        f"{category:25s} "
        f"{count:6d} "
        f"({percentage:5.2f}%)"
    )

# ---------------------------------------------------------
# Sample messages per category
# ---------------------------------------------------------

print("\n========== EXAMPLES ==========")

for category, keywords in categories.items():

    print(f"\n\n### {category}")

    found = 0

    for _, row in df.iterrows():

        text = str(row["customer_text"]).lower()

        if any(keyword in text for keyword in keywords):

            print("\nCustomer:")
            print(row["customer_text"])

            print("Support:")
            print(row["support_text"])

            found += 1

            if found >= 5:
                break

print("\nAnalysis completed.")