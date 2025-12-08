import pandas as pd
from typing import Tuple, Iterable, Optional
from .paths import find_dataset_dir, list_data_files

def load_columns() -> Tuple[str, list]:
    ds = find_dataset_dir()
    cols = pd.read_csv(f"{ds}/data_names.csv", header=None).iloc[0].tolist()
    return ds, cols

def _detect_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "Label", "attack", "category", "class"]:
        if c in df.columns:
            return c
    raise ValueError("No label column found.")

def _map_labels(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        return (
            series.astype(str).str.lower().map({
                "normal": 0, "benign": 0,
                "attack": 1, "anomaly": 1,
                "ddos": 1, "dos": 1, "theft": 1,
                "scan": 1, "keylogging": 1, "data_exfiltration": 1
            }).fillna(1).astype(int)
        )
    s = pd.to_numeric(series, errors="coerce").fillna(1).astype(int)
    return s.clip(0, 1)

def _iter_chunks(csv_path: str, names: list, chunksize: int) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(csv_path, names=names, chunksize=chunksize, low_memory=False):
        yield chunk

def load_sample(
    target_per_class: int = 5000,
    chunksize: int = 100_000,
    max_files: Optional[int] = None,
    allow_partial: bool = True
):
    """
    Stream all data_*.csv files and collect a balanced sample.
    - target_per_class: desired rows for each class (0 and 1)
    - chunksize: rows per chunk while streaming
    - max_files: limit files scanned (None = scan all)
    - allow_partial: if minority < target, use all minority and downsample majority to match
    """
    ds, cols = load_columns()
    files = list_data_files(ds)
    if max_files is not None:
        files = files[:max_files]

    keep0, keep1 = [], []
    rows0, rows1 = 0, 0
    label_col_detected: Optional[str] = None

    for i, f in enumerate(files, start=1):
        print(f"📂  Scanning file {i}/{len(files)}: {f}")
        for chunk in _iter_chunks(f, names=cols, chunksize=chunksize):
            if label_col_detected is None:
                label_col_detected = _detect_label_col(chunk)
            y = _map_labels(chunk[label_col_detected])
            chunk = chunk.assign(__y__=y)

            need0 = max(target_per_class - rows0, 0)
            need1 = max(target_per_class - rows1, 0)

            if need0 > 0:
                take0 = chunk[chunk["__y__"] == 0].head(need0)
                if not take0.empty:
                    keep0.append(take0)
                    rows0 += len(take0)

            if need1 > 0:
                take1 = chunk[chunk["__y__"] == 1].head(need1)
                if not take1.empty:
                    keep1.append(take1)
                    rows1 += len(take1)

            print(f"   → collected so far: class0={rows0}, class1={rows1}")

            if rows0 >= target_per_class and rows1 >= target_per_class:
                break
        if rows0 >= target_per_class and rows1 >= target_per_class:
            break

    df0 = pd.concat(keep0, ignore_index=True) if keep0 else pd.DataFrame()
    df1 = pd.concat(keep1, ignore_index=True) if keep1 else pd.DataFrame()

    if len(df0) == 0 or len(df1) == 0:
        raise RuntimeError(
            f"❌ Could not collect any rows for one of the classes. "
            f"Collected: class0={len(df0)}, class1={len(df1)}. "
            f"Try reducing chunksize, scanning more files (max_files=None), "
            f"or verify the label mapping."
        )

    # Balance (or partial-balance) the sample
    if len(df0) < target_per_class or len(df1) < target_per_class:
        if not allow_partial:
            raise RuntimeError(
                f"Not enough rows: class0={len(df0)}, class1={len(df1)}. "
                "Increase scan or set allow_partial=True."
            )
        minority = min(len(df0), len(df1))
        print(f"⚠️ Partial balance: class0={len(df0)}, class1={len(df1)}. Using {minority} per class.")
        df0 = df0.iloc[:minority]
        df1 = df1.iloc[:minority]
    else:
        df0 = df0.iloc[:target_per_class]
        df1 = df1.iloc[:target_per_class]

    # Combine & shuffle
    df_bal = pd.concat([df0, df1], ignore_index=True).sample(frac=1.0, random_state=42)

    # ✅ Create canonical numeric label and drop confusing originals
    df_bal["label"] = df_bal["__y__"].astype(int)
    for c in ["__y__", "attack", "Attack", "Label", "label.1", "category", "class"]:
        if c in df_bal.columns and c != "label":
            df_bal.drop(columns=c, inplace=True, errors="ignore")

    out = f"{ds}/BotIoT_balanced_{len(df0)}_{len(df1)}.csv"
    df_bal.to_csv(out, index=False)
    print(f"✅ Balanced sample built: class0={len(df0)}, class1={len(df1)} → saved {out}")
    return df_bal, out
