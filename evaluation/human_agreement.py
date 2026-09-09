import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compare(human_csv, judge_csv, column="correctness"):
    human = pd.read_csv(human_csv)
    judge = pd.read_csv(judge_csv)

    merged = human.merge(judge, on="id", suffixes=("_human", "_judge"))
    a = merged[f"{column}_human"]
    b = merged[f"{column}_judge"]

    print("Exact agreement:", float((a == b).mean()))
    print("Cohen kappa:", float(cohen_kappa_score(a, b)))

if __name__ == "__main__":
    print("Provide human and judge CSVs to compare ratings.")
