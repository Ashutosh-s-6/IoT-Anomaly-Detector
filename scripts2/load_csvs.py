# load_csvs.py  (RAM-safe, uses chunking + sampling)
import os
import glob
import pandas as pd

# folder with all CTU IoT-23 CSVs
DATA_DIR = r"C:\IoT-Anomaly-Detector\datasets\Open"
# or relative: DATA_DIR = "datasets/Open"
OUT_FILE = r"C:\IoT-Anomaly-Detector\datasets\iot23_raw.parquet"
# or relative: OUT_FILE = "datasets/iot23_raw.parquet"

def load_all_csvs(sample_frac=0.05, chunksize=200_000):
    """
    sample_frac = fraction of rows to keep from each chunk (0.05 = 5%)
    chunksize   = rows per chunk to load at once
    """
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"[+] Found {len(files)} CSV files in {DATA_DIR}")

    if not files:
        print("[!] No CSV files found. Check DATA_DIR path.")
        return

    df_list = []

    for f in files:
        print(f"[+] Loading in chunks from: {f}")
        reader = pd.read_csv(
            f,
            on_bad_lines='skip',
            low_memory=False,
            chunksize=chunksize
        )

        for i, chunk in enumerate(reader):
            # take a random sample from each chunk to avoid blowing RAM
            if 0 < sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac, random_state=42)

            df_list.append(chunk)
            print(f"    - processed chunk {i+1}")

    print(f"[+] Concatenating {len(df_list)} sampled chunks...")
    df = pd.concat(df_list, ignore_index=True)

    print(f"[+] Saving to Parquet → {OUT_FILE}")
    df.to_parquet(OUT_FILE)
    print("[✓] Done. Sample of data:")
    print(df.head())

if __name__ == "__main__":
    load_all_csvs()
