# Hiver SDE Intern — AI Customer Support Agent

A reproducible starter implementation for the Hiver SDE Intern take-home assignment.

## What this project does

The system is designed for one selected brand from the Customer Support on Twitter dataset:

1. Discover/define a small intent taxonomy from the selected brand's data.
2. Classify incoming customer messages.
3. Retrieve similar historical support cases.
4. Generate a grounded support reply using historical resolutions.
5. Decide `AUTO-HANDLE` vs `ESCALATE`, with a reason.
6. Evaluate against a hand-labelled golden set and compare against baselines.

> Important: do not report fabricated results. Run the pipeline on the actual dataset and golden set before filling the report.

## Repository structure

```text
data/
  raw/                 # Put the Kaggle dataset here
  processed/           # Generated processed data
  golden/              # Hand-labelled evaluation set

src/
  config.py
  data_loader.py
  preprocessing.py
  conversation_builder.py
  intent_classifier.py
  retrieval.py
  response_generator.py
  escalation.py
  agent.py
  evaluation.py

baselines/
  majority.py
  tfidf.py

evaluation/
  run_eval.py
  judge.py
  human_agreement.py

reports/
  report.md

tests/
  test_agent.py
```

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Optional LLM support:

```bash
set OPENAI_API_KEY=your_key_here
```

Linux/macOS:

```bash
export OPENAI_API_KEY=your_key_here
```

The default code has a deterministic fallback response generator so the pipeline can be tested without an API key.

## Dataset

Download the Kaggle `thoughtvector/customer-support-on-twitter` dataset and place the relevant CSV in:

```text
data/raw/
```

The loader attempts to detect common column names automatically, but you should inspect the dataset before final evaluation.

## Recommended workflow

### 1. Inspect the dataset

```bash
python -m src.data_loader --input data/raw
```

This prints file/column information and brand frequencies.

### 2. Build processed conversations

```bash
python -m src.conversation_builder --input data/raw --output data/processed/conversations.jsonl
```

### 3. Build a development set

Use actual selected-brand examples and label the intent taxonomy after inspecting the data.

### 4. Run baselines

```bash
python -m baselines.majority
python -m baselines.tfidf
```

### 5. Run the agent

```bash
python -m src.agent --message "I was charged twice for my order"
```

### 6. Evaluate

After creating `data/golden/golden_set.csv`:

```bash
python -m evaluation.run_eval
```

## Golden set

Target 150–250 manually labelled examples (recommended: 200).

Required fields:

```text
id,message,true_intent,true_escalation
```

Keep the golden set separate from development/training data.

## Evaluation

At minimum report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-intent F1
- Confusion matrix
- Retrieval Recall@1/@3/@5
- Escalation precision/recall/F1
- False auto-handling rate
- LLM-judge scores
- Human-vs-judge agreement

## Important limitations

This starter repository does not know the selected brand, final intent taxonomy, actual dataset statistics, or measured results until the real dataset is supplied and processed. Those values must be generated from the data rather than invented.

## Hiver deliverables checklist

- [ ] Runnable repository
- [ ] 150–250 example golden evaluation set
- [ ] Sampling/labeling note
- [ ] Automated evaluation harness
- [ ] LLM-as-judge rubric
- [ ] Human agreement evidence
- [ ] Two baselines
- [ ] Top five failure modes with real examples
- [ ] “What is misleading about my headline number?”
- [ ] One-week-next-steps section
- [ ] 10–15 decision log entries
- [ ] Report ≤ 6 pages / README section
