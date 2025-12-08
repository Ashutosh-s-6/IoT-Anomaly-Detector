import pandas as pd
import numpy as np

INPUT = r"C:/IoT-Anomaly-Detector/datasets/iot23_raw.parquet"
OUTPUT = r"C:/IoT-Anomaly-Detector/datasets/iot23_clean.parquet"

def label_from_tunnel(val):
    if pd.isna(val):
        return 0
    val = str(val).lower()
    if "benign" in val:
        return 0
    if any(x in val for x in ["malicious", "okiru", "gafgyt", "mirai", "c&c", "ddos", "scan"]):
        return 1
    return 0

def preprocess():
    print("[+] Loading IoT-23 raw data...")
    df = pd.read_parquet(INPUT)

    print("[+] Creating attack labels from tunnel_parents...")
    df["attack"] = df["tunnel_parents"].apply(label_from_tunnel)

    cols_to_drop = ["uid","id.orig_h","id.resp_h","proto","service",
                    "label","detailed-label","tunnel_parents"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    print("[+] Saving cleaned dataset...")
    df.to_parquet(OUTPUT)
    print(f"[✓] Saved → {OUTPUT}")
    print(df.head())
    print(df['attack'].value_counts())

if __name__ == "__main__":
    preprocess()
