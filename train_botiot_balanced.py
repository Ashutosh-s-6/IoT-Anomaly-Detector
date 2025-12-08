# Finan Model training script for Bot-IoT dataset with balanced sampling
import os, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error
)

# ---------------- CONFIG ----------------
DATASETS_DIR = Path("datasets")         # has data_1.csv ... data_74.csv and data_names.csv
DATA_NAMES_FILE = DATASETS_DIR / "data_names.csv"
DATA_GLOB = "data_*.csv"

SEED = 42

FILES_PER_BATCH = 5


SAMPLE_FRAC_PER_BATCH = 0.20   


MAX_ROWS_PER_CLASS = 300_000   # you can lower to 150_000 if still heavy

# columns to drop (IPs, IDs, timestamps, etc.)
DROP_COLS = [
    "pkSeqID", "stime", "ltime", "seq",
    "saddr", "daddr", "sport", "dport",
    "smac", "dmac", "soui", "doui",
    "src_ip", "dst_ip", "timestamp", "flow_id"
]

# label candidates
LABEL_CANDIDATES = ["label", "Label", "attack", "class", "category"]

# likely categorical columns
CATEGORICAL_GUESS = ["proto", "state", "service", "flgs"]

BACKEND_DIR = Path("backend")
BACKEND_DIR.mkdir(exist_ok=True)
MODEL_OUT = BACKEND_DIR / "BotIotmodel.pkl"

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# ---------------- HELPERS ----------------
def load_columns():
    if not DATA_NAMES_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_NAMES_FILE}")
    cols = pd.read_csv(DATA_NAMES_FILE, header=None).iloc[0].tolist()
    return [c.strip() for c in cols]


def detect_header_in_parts(first_csv_path: str, columns: list[str]) -> bool:
    with open(first_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    return (
        [x.strip().lower() for x in first_line.split(",")]
        == [x.strip().lower() for x in columns]
    )


def normalize_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0, "1": 1,
        "normal": 0, "benign": 0,
        "attack": 1, "attacks": 1, "anomaly": 1, "malicious": 1,
        "ddos": 1, "dos": 1, "theft": 1,
        "scan": 1, "keylogging": 1, "data_exfiltration": 1,
        "true": 1, "yes": 1, "false": 0, "no": 0,
    }
    mapped = s.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.astype("float64")
    out = out.where(~out.isna(), numeric)
    keep = ~out.isna()
    out = out[keep].astype(int).clip(0, 1)
    return out, keep


def find_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise RuntimeError(f"No label column found. Tried: {LABEL_CANDIDATES}")


def collect_balanced_sample(columns: list[str]) -> tuple[pd.DataFrame, str]:
    """
    MEMORY-SAFE:
      - Read a few files at a time (FILES_PER_BATCH)
      - From each batch, take a small fraction per class
      - Keep two pools: normals and attacks, capped at MAX_ROWS_PER_CLASS
    """
    paths = sorted(glob.glob(str(DATASETS_DIR / DATA_GLOB)))
    if not paths:
        raise RuntimeError(f"No files matching {DATA_GLOB} in {DATASETS_DIR}")

    print(f"[i] Found {len(paths)} Bot-IoT CSV parts")

    has_header = detect_header_in_parts(paths[0], columns)
    print(f"[i] Part files already have header? {has_header}")

    norm_pool = []
    att_pool = []
    norm_count = 0
    att_count = 0

    for i in range(0, len(paths), FILES_PER_BATCH):
        batch_paths = paths[i:i+FILES_PER_BATCH]
        print(f"[i] Reading batch {i//FILES_PER_BATCH+1} with {len(batch_paths)} files")

        dfs = []
        for p in batch_paths:
            part = pd.read_csv(
                p,
                names=columns,
                header=None,
                skiprows=1 if has_header else 0,
                low_memory=False
            )
            dfs.append(part)

        df_batch = pd.concat(dfs, ignore_index=True)
        print(f"    - batch shape before label cleaning: {df_batch.shape}")

        label_col = find_label_col(df_batch)
        y_batch, keep = normalize_label(df_batch[label_col])
        df_batch = df_batch.loc[keep].reset_index(drop=True)
        y_batch = y_batch.reset_index(drop=True)
        df_batch[label_col] = y_batch

        print(f"    - batch shape after label cleaning: {df_batch.shape}")

        # split by class
        df_norm = df_batch[y_batch == 0]
        df_att = df_batch[y_batch == 1]

        # sample some normals if we still need them
        need_norm = MAX_ROWS_PER_CLASS - norm_count
        if need_norm > 0 and len(df_norm):
            take_n = int(len(df_norm) * SAMPLE_FRAC_PER_BATCH)
            if take_n <= 0:
                take_n = min(len(df_norm), need_norm)
            else:
                take_n = min(take_n, need_norm)
            sample_norm = df_norm.sample(n=take_n, random_state=SEED)
            norm_pool.append(sample_norm)
            norm_count += len(sample_norm)
            print(f"    - added normals: {len(sample_norm):,} (total normals: {norm_count:,})")

        # sample some attacks if we still need them
        need_att = MAX_ROWS_PER_CLASS - att_count
        if need_att > 0 and len(df_att):
            take_a = int(len(df_att) * SAMPLE_FRAC_PER_BATCH)
            if take_a <= 0:
                take_a = min(len(df_att), need_att)
            else:
                take_a = min(take_a, need_att)
            sample_att = df_att.sample(n=take_a, random_state=SEED)
            att_pool.append(sample_att)
            att_count += len(sample_att)
            print(f"    - added attacks: {len(sample_att):,} (total attacks: {att_count:,})")

        # if we have enough for both classes, we can stop early
        if norm_count >= MAX_ROWS_PER_CLASS and att_count >= MAX_ROWS_PER_CLASS:
            print("[i] Reached MAX_ROWS_PER_CLASS for both classes. Stopping early.")
            break

    if not norm_pool or not att_pool:
        raise RuntimeError("Not enough data collected for one or both classes.")

    df_norm_all = pd.concat(norm_pool, ignore_index=True)
    df_att_all = pd.concat(att_pool, ignore_index=True)

    print(f"[i] Pooled normals: {df_norm_all.shape}, pooled attacks: {df_att_all.shape}")

    # FINAL BALANCE: keep min(class sizes) from both
    label_col = find_label_col(df_norm_all)  # same name as in batch
    n_norm = len(df_norm_all)
    n_att = len(df_att_all)
    target = min(n_norm, n_att)
    print(f"[i] Final balance target per class: {target:,} (normals={n_norm:,}, attacks={n_att:,})")

    df_norm_bal = df_norm_all.sample(n=target, random_state=SEED)
    df_att_bal = df_att_all.sample(n=target, random_state=SEED)

    df_bal = pd.concat([df_norm_bal, df_att_bal], ignore_index=True)
    df_bal = df_bal.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"[i] Balanced dataset shape: {df_bal.shape}")
    print(f"[i] Balanced class counts: {df_bal[label_col].value_counts().to_dict()}")

    return df_bal, label_col


def build_features(df: pd.DataFrame, label_col: str):
    drop_actual = [c for c in DROP_COLS if c in df.columns]

    X = df.drop(columns=drop_actual + [label_col], errors="ignore").copy()
    y = df[label_col].astype(int).values

    cat_cols = [c for c in CATEGORICAL_GUESS if c in X.columns]

    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print(f"[i] Dropping all-NaN columns: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)

    X = X.fillna(0)

    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"[i] Final features: {len(X.columns)} "
          f"(numeric={len(num_cols)}, categorical={len(cat_cols)})")
    return X, y, num_cols, cat_cols, drop_actual


def build_pipeline(num_cols, cat_cols):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        n_jobs=-1,
        random_state=SEED,
        class_weight=None  # data already balanced
    )

    pipe = Pipeline([
        ("prep", pre),
        ("rf", clf),
    ])
    return pipe


# ---------------- MAIN ----------------
def main():
    print("[i] Loading column names from data_names.csv …")
    cols = load_columns()

    print("[i] Collecting balanced sample in memory-safe way …")
    df_bal, label_col = collect_balanced_sample(cols)

    print("[i] Preparing features …")
    X, y, num_cols, cat_cols, drop_actual = build_features(df_bal, label_col)

    print("[i] Train/test split (stratified) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    print(f"[i] X_train={X_train.shape}, X_test={X_test.shape}")

    print("[*] Building pipeline and training …")
    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)
    print("[OK] Model trained")

    print("[i] Evaluating on test set …")
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "support": {"0": int((y_test == 0).sum()), "1": int((y_test == 1).sum())},
    }

    if proba is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["auc"] = float("nan")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / (fp + tn) if (fp + tn) else 0.0)
        metrics["tnr_specificity"] = float(tn / (tn + fp) if (tn + fp) else 0.0)

    print(json.dumps(metrics, indent=2))

    bundle = {
        "pipeline": pipe,
        "label_col": label_col,
        "drop_cols": drop_actual,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "columns_from_names": cols,
        "feature_cols": num_cols + cat_cols,
    }

    joblib.dump(bundle, MODEL_OUT)
    size_mb = MODEL_OUT.stat().st_size / 1e6
    print(f"[OK] Saved model to {MODEL_OUT} ({size_mb:.2f} MB)")

    (REPORT_DIR / "metrics_BotIot_balanced.json").write_text(json.dumps(metrics, indent=2))
    print("[OK] Metrics saved to reports/metrics_BotIot_balanced.json")


if __name__ == "__main__":
    main()
