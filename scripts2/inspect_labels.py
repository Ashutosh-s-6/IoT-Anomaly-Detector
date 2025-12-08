# inspect_labels.py
import pandas as pd
import os

INPUT = r"C:\IoT-Anomaly-Detector\datasets\iot23_raw.parquet"

def main():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"File not found: {INPUT}")

    print(f"[+] Loading {INPUT} ...")
    df = pd.read_parquet(INPUT)

    print("[+] Columns:", list(df.columns))

    # Show some samples of label columns if they exist
    for col in ["label", "detailed-label", "Label", "LabelName"]:
        if col in df.columns:
            print(f"\n[+] Value counts for '{col}' (top 30):")
            print(df[col].value_counts(dropna=False).head(30))

if __name__ == "__main__":
    main()
