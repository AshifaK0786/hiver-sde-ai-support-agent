from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(".")
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

RESULTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

gold_path = DATA / "golden" / "verizon_golden_200_review.csv"

if not gold_path.exists():
    print("ERROR: Golden file not found:", gold_path)
    raise SystemExit(1)

df = pd.read_csv(gold_path)

# =========================================================
# GOLD LABEL STATUS
# =========================================================

if "gold_intent" in df.columns:
    gold = df["gold_intent"].fillna("").astype(str).str.strip()
else:
    gold = pd.Series([""] * len(df))

verified = pd.Series([False] * len(df))

if "human_verified" in df.columns:
    verified = df["human_verified"].astype(str).str.lower().isin(
        ["true", "yes", "1", "verified"]
    )

if verified.sum() == len(df) and (gold != "").all():
    evaluation_labels = gold
    label_status = "HUMAN-VERIFIED"
else:
    if "suggested_gold_intent" in df.columns:
        evaluation_labels = df["suggested_gold_intent"].fillna(
            df["candidate_intent"]
        ).astype(str)
    else:
        evaluation_labels = df["candidate_intent"].fillna("unknown").astype(str)

    label_status = "PROVISIONAL - AI/weak labels, NOT human verified"

df["evaluation_intent"] = evaluation_labels

# =========================================================
# CLASSIFIER
# =========================================================

model = joblib.load(MODELS / "intent_classifier.joblib")

texts = df["customer_text"].fillna("").astype(str).tolist()

predictions = model.predict(texts)

df["predicted_intent"] = predictions

# =========================================================
# OUTAGE SAFETY RULE
# =========================================================

outage_signals = [
    "is verizon down",
    "verizon is down",
    "everything is down",
    "nothing is working",
    "all services are down",
    "service outage",
    "there is an outage",
    "there's an outage",
    "all of verizon is down"
]

for i, text in enumerate(texts):
    text_lower = text.lower()

    if any(signal in text_lower for signal in outage_signals):
        df.loc[i, "predicted_intent"] = "service_outage"

# =========================================================
# INTENT METRICS
# =========================================================

y_true = df["evaluation_intent"]
y_pred = df["predicted_intent"]

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0
)

# =========================================================
# MAJORITY BASELINE
# =========================================================

majority_class = y_true.value_counts().index[0]

majority_predictions = [majority_class] * len(y_true)

majority_accuracy = accuracy_score(
    y_true,
    majority_predictions
)

majority_f1 = f1_score(
    y_true,
    majority_predictions,
    average="macro",
    zero_division=0
)

# =========================================================
# SAVE INTENT RESULTS
# =========================================================

pd.DataFrame([
    {
        "system": "Majority baseline",
        "accuracy": majority_accuracy,
        "macro_f1": majority_f1
    },
    {
        "system": "TF-IDF + Logistic Regression + outage rule",
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }
]).to_csv(
    RESULTS / "intent_metrics.csv",
    index=False
)

# =========================================================
# PER INTENT RESULTS
# =========================================================

rows = []

for label, values in report.items():

    if isinstance(values, dict):

        rows.append({
            "intent": label,
            "precision": values.get("precision", 0),
            "recall": values.get("recall", 0),
            "f1": values.get("f1-score", 0),
            "support": values.get("support", 0)
        })

pd.DataFrame(rows).to_csv(
    RESULTS / "per_intent_metrics.csv",
    index=False
)

# =========================================================
# SAVE PREDICTIONS
# =========================================================

output_columns = [
    "customer_tweet_id",
    "customer_text",
    "evaluation_intent",
    "predicted_intent"
]

if "candidate_intent" in df.columns:
    output_columns.append("candidate_intent")

df[output_columns].to_csv(
    RESULTS / "intent_predictions.csv",
    index=False
)

# =========================================================
# ESCALATION POLICY
# =========================================================

sensitive_intents = {
    "billing_payment",
    "account_authentication",
    "order_equipment",
    "plan_upgrade"
}

def should_escalate(intent):
    return (
        intent in sensitive_intents
        or intent == "service_outage"
    )

df["expected_escalate"] = df["evaluation_intent"].apply(
    should_escalate
)

df["predicted_escalate"] = df["predicted_intent"].apply(
    should_escalate
)

tp = (
    (df["expected_escalate"] == True) &
    (df["predicted_escalate"] == True)
).sum()

fp = (
    (df["expected_escalate"] == False) &
    (df["predicted_escalate"] == True)
).sum()

fn = (
    (df["expected_escalate"] == True) &
    (df["predicted_escalate"] == False)
).sum()

tn = (
    (df["expected_escalate"] == False) &
    (df["predicted_escalate"] == False)
).sum()

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

escalation_f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall
    else 0
)

auto_handle_rate = (
    (df["predicted_escalate"] == False).mean()
)

pd.DataFrame([{
    "precision": precision,
    "recall": recall,
    "f1": escalation_f1,
    "true_positive": tp,
    "false_positive": fp,
    "false_negative": fn,
    "true_negative": tn,
    "auto_handle_rate": auto_handle_rate
}]).to_csv(
    RESULTS / "escalation_metrics.csv",
    index=False
)

# =========================================================
# FAILURE EXAMPLES
# =========================================================

errors = df[
    df["evaluation_intent"] != df["predicted_intent"]
].copy()

errors[
    [
        "customer_tweet_id",
        "customer_text",
        "evaluation_intent",
        "predicted_intent"
    ]
].head(20).to_csv(
    RESULTS / "failure_examples.csv",
    index=False
)

# =========================================================
# FINAL REPORT
# =========================================================

report_text = f"""# Hiver SDE Intern Assessment
## Verizon AI Support Agent

## 1. Problem

The system provides an AI support agent for VerizonSupport.

Given a customer message, it:

1. Classifies the issue into an operational intent.
2. Retrieves similar historical customer-support interactions.
3. Uses historical responses as grounding evidence.
4. Decides whether to auto-handle or escalate to a human.
5. Drafts a response.

## 2. Intent taxonomy

Seven operational intents were selected:

- internet_connectivity
- service_outage
- billing_payment
- account_authentication
- order_equipment
- plan_upgrade
- general_support

## 3. Architecture

Customer message
â†’ TF-IDF + Logistic Regression
â†’ outage safety correction
â†’ intent-filtered retrieval
â†’ similarity threshold
â†’ escalation decision
â†’ grounded response

## 4. Baselines

### Baseline 1 â€” Majority classifier

Accuracy: {majority_accuracy:.4f}

Macro F1: {majority_f1:.4f}

### Baseline 2 â€” TF-IDF + Logistic Regression

Accuracy: {accuracy:.4f}

Macro F1: {macro_f1:.4f}

## 5. Evaluation

Evaluation examples: {len(df)}

Label status:

{label_status}

**Important:** if the golden examples have not been independently
human verified, these numbers must be treated as provisional and
must not be described as human-evaluated performance.

## 6. Escalation

The system conservatively escalates:

- billing/payment issues
- account authentication issues
- order/equipment issues
- plan upgrades
- service outages

Current evaluation:

Precision: {precision:.4f}

Recall: {recall:.4f}

F1: {escalation_f1:.4f}

Auto-handle rate: {auto_handle_rate:.4f}

## 7. Failure modes

### Failure 1 â€” Internet connectivity vs outage

Example:

"Is Verizon internet down? Nothing is working"

The classifier can focus on the word "internet" even when the customer
is reporting a broad outage.

Mitigation:

A rule-based outage correction was added.

### Failure 2 â€” Ambiguous customer messages

Short messages may not contain enough information to distinguish
between multiple intents.

### Failure 3 â€” Billing/account overlap

Terms such as account, login, bill and payment can appear in multiple
contexts.

### Failure 4 â€” Weak historical retrieval

Some messages have no sufficiently similar historical support example.

Mitigation:

Low retrieval similarity causes escalation.

### Failure 5 â€” Verbatim historical response reuse

Copying a historical support response can preserve outdated context,
agent handles or assumptions.

A future version should generate a new response from multiple retrieved
examples instead.

## 8. What is misleading about my headline number?

The classifier's weak-label score can appear stronger than actual
production performance because the training labels were initially
generated using heuristic/weak supervision.

Therefore the weak-label holdout score should not be interpreted as
independent human-verified accuracy.

The golden set is intended to provide a stronger evaluation once
human annotation is completed.

## 9. One-more-week plan

1. Complete independent annotation of all 200 golden examples.
2. Add a second annotator.
3. Calculate Cohen's kappa for human agreement.
4. Add LLM-as-judge evaluation for response correctness,
   groundedness, helpfulness and safety.
5. Replace verbatim historical response reuse with grounded LLM
   generation.
6. Reconstruct complete multi-turn conversations.
7. Add retrieval Recall@1/3/5.
8. Calibrate confidence and optimize escalation thresholds.

## 10. Decision log

1. Selected VerizonSupport because of sufficient support interaction volume.
2. Defined seven operational intents.
3. Used weak supervision for initial training.
4. Kept the golden set separate from training.
5. Used TF-IDF + Logistic Regression as an interpretable baseline.
6. Added a majority-class baseline.
7. Filtered retrieval candidates by predicted intent.
8. Added explicit outage detection.
9. Used conservative escalation for sensitive intents.
10. Added a retrieval similarity threshold.
11. Used historical support responses as grounding evidence.
12. Avoided presenting weak-label accuracy as human accuracy.
13. Preserved ambiguous examples for evaluation.
14. Identified verbatim response reuse as a limitation.
15. Prioritized safe escalation over aggressive automation.

## 11. Evaluation artifacts

Generated files:

- results/intent_metrics.csv
- results/per_intent_metrics.csv
- results/intent_predictions.csv
- results/escalation_metrics.csv
- results/failure_examples.csv

## 12. Reproducibility

Main agent:

```text
"""
