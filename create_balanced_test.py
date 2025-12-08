import os, glob, argparse, random
from pathlib import Path
import pandas as pd
import numpy as np

def detect_header(csv_path, expected_columns):
    """Return True if first row matches the expected header."""
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.readline().strip().split(",")
        if expected_columns is None:
            return False
        return [h.strip().lower() for h in head] == [c.strip().lower() for c in expected_columns]
    except Exception:
        return False

def normalize_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Map diverse label encodings to {0,1}. Return (y, mask_kept)."""
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "attack": 1, "attacks": 1, "anomaly": 1, "malicious": 1, "botnet": 1,
        "true": 1, "yes": 1,
        "normal": 0, "benign": 0, "false": 0, "no": 0
    }
    out = s.map(mapping)
    # numeric fallback
    num = pd.to_numeric(series, errors="coerce")
    out = out.where(~out.isna(), num)
    keep = ~out.isna()
    return out.loc[keep].astype(int), keep

def read_part(path, expected_columns=None):
    """Read a CSV part with header detection + robust dtype handling."""
    has_header = detect_header(path, expected_columns)
    df = pd.read_csv(
        path,
        names=expected_columns if expected_columns is not None else None,
        header=None if expected_columns is not None else (0 if has_header else "infer"),
        low_memory=False
    )
    return df

def main():
    ap = argparse.ArgumentParser("Create a balanced/specified-ratio test CSV from many data_*.csv files")
    ap.add_argument("--datasets-dir", default="datasets", help="Folder containing data_*.csv and optional data_names.csv")
    ap.add_argument("--pattern", default="data_*.csv", help="Glob for input files")
    ap.add_argument("--rows", type=int, default=100_000, help="Total rows in the balanced test file")
    ap.add_argument("--pos-frac", type=float, default=0.5, help="Fraction of attacks (1s) in output")
    ap.add_argument("--label-col", default="attack", help="Label column name if present")
    ap.add_argument("--out", default="datasets/balanced_test.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    datasets_dir = Path(args.datasets_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load canonical column list if available (keeps schema consistent)
    names_path = datasets_dir / "data_names.csv"
    expected_columns = None
    if names_path.exists():
        try:
            expected_columns = pd.read_csv(names_path, header=None).iloc[0].tolist()
            print(f"[i] Loaded {len(expected_columns)} columns from data_names.csv")
        except Exception:
            expected_columns = None

    files = sorted(glob.glob(str(datasets_dir / args.pattern)))
    if not files:
        raise SystemExit(f"No files found for {datasets_dir}/{args.pattern}")

    total_needed = int(args.rows)
    need_pos = int(round(total_needed * args.pos_frac))
    need_neg = total_needed - need_pos

    label_candidates = [args.label_col, "attack", "label", "Category", "category", "subcategory"]

    pos_parts, neg_parts = [], []
    seen = 0

    for fp in files:
        df = read_part(fp, expected_columns=expected_columns)
        # find a label column
        label_col = None
        for c in label_candidates:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            # If there is no label column, skip this file
            print(f"[!] Skipping (no label column): {fp}")
            continue

        y, keep = normalize_label(df[label_col])
        df = df.loc[keep].reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Split this part by class
        pos_df = df[y == 1]
        neg_df = df[y == 0]

        # sample up to what we still need
        if need_pos > 0 and len(pos_df) > 0:
            take = min(len(pos_df), need_pos)
            pos_parts.append(pos_df.sample(take, random_state=args.seed))
            need_pos -= take

        if need_neg > 0 and len(neg_df) > 0:
            take = min(len(neg_df), need_neg)
            neg_parts.append(neg_df.sample(take, random_state=args.seed))
            need_neg -= take

        seen += len(df)
        print(f"[i] {os.path.basename(fp)} | kept={len(df):,} | need_pos={need_pos:,}, need_neg={need_neg:,}")

        if need_pos <= 0 and need_neg <= 0:
            break

    if need_pos > 0 or need_neg > 0:
        print(f"[!] Not enough rows to hit the exact target. Missing -> pos:{need_pos}, neg:{need_neg}")

    out_df = pd.concat(pos_parts + neg_parts, ignore_index=True)

    # Shuffle for good measure
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Write
    out_df.to_csv(out_path, index=False)
    # Quick summary
    # Try to recompute normalized counts from the saved file for a clear report
    y2, keep2 = normalize_label(out_df[label_candidates[0]] if label_candidates[0] in out_df.columns else out_df[ [c for c in label_candidates if c in out_df.columns][0] ])
    pos_count = int((y2 == 1).sum())
    neg_count = int((y2 == 0).sum())

    print("\n[OK] Balanced test file created.")
    print(f" -> Path: {out_path}")
    print(f" -> Shape: {out_df.shape}")
    print(f" -> Class counts: 1 (attack) = {pos_count:,} | 0 (normal) = {neg_count:,}")

if __name__ == "__main__":
    main()
