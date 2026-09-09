import re
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_dataframe(df, text_col):
    out = df.copy()
    out["clean_text"] = out[text_col].map(clean_text)
    out = out[out["clean_text"].str.len() > 0]
    out = out.drop_duplicates(subset=["clean_text"])
    return out.reset_index(drop=True)
