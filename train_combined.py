# train_combined.py — balanced + memory-safe + full-dataset training
import os, glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error
)

# ---------------- CONFIG ----------------
DATASETS_DIR = "datasets"
BATCH = 5
SAMPLE_FRAC = 0.30         # take 30% rows per batch to avoid memory crash
MAX_ROWS = 12_000_000      # final cap for combined sample
SEED = 42

REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_DIR = Path("backend"); BACKEND_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = BACKEND_DIR / "model.pkl"

DROP_COLS = ['pkSeqID','stime','ltime','seq','saddr','daddr','sport','dport','smac','dmac','soui','doui']
LABEL_CANDIDATES = ["attack","label"]

# ---------------- HELPERS ----------------
def load_columns(datasets_dir: str):
    names = pd.read_csv(os.path.join(datasets_dir, "data_names.csv"), header=None).iloc[0].tolist()
    return [c.strip() for c in names]

def normalize_label(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    m = {"1":1,"0":0,"attack":1,"attacks":1,"anomaly":1,"malicious":1,
         "true":1,"yes":1,"normal":0,"benign":0,"false":0,"no":0}
    out = s.map(m)
    num = pd.to_numeric(s, errors="coerce")
    out = out.where(~out.isna(), num)
    return out.astype("Int64")

def collect_sample(datasets_dir: str, cols):
    files = sorted(glob.glob(os.path.join(datasets_dir, "data_*.csv")))
    if not files:
        raise RuntimeError("No data_*.csv files found in datasets/")

    pool = []
    for i in range(0, len(files), BATCH):
        batch = files[i:i+BATCH]
        df_part = pd.concat(
            (pd.read_csv(p, names=cols, header=0, low_memory=False) for p in batch),
            ignore_index=True
        )

        label_col = None
        for cand in LABEL_CANDIDATES:
            if cand in df_part.columns:
                label_col = cand
                break
        if label_col is None:
            continue

        y = normalize_label(df_part[label_col])
        keep = ~y.isna()
        df_part = df_part.loc[keep].reset_index(drop=True)
        df_part[label_col] = y.loc[keep].astype(int)
        df_part = df_part.drop_duplicates().fillna(0)

        # sample fraction of each chunk to prevent memory overflow
        df_part = df_part.sample(frac=SAMPLE_FRAC, random_state=SEED)
        pool.append(df_part)
        print(f"[+] Sampled {len(df_part):,} rows from chunk {(i//BATCH)+1}")

    if not pool:
        raise RuntimeError("No labeled rows found.")
    df = pd.concat(pool, ignore_index=True)
    print(f"[i] Combined sample shape before cap: {df.shape}")

    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
        print(f"[i] Combined sample shape after cap:  {df.shape}")

    return df

def choose_features(df: pd.DataFrame, label_col: str):
    X = df.drop(columns=[label_col] + [c for c in DROP_COLS if c in df.columns], errors="ignore")
    y = df[label_col].astype(int).values

    # downcast numeric columns to save memory
    for c in X.columns:
        if X[c].dtype == object:
            num = pd.to_numeric(X[c], errors="coerce")
            if num.notna().any():
                if (num.dropna() % 1 == 0).all():
                    X[c] = num.astype("Int32")
                else:
                    X[c] = num.astype("float32")
        elif np.issubdtype(X[c].dtype, np.floating):
            X[c] = X[c].astype("float32")
        elif np.issubdtype(X[c].dtype, np.integer):
            X[c] = X[c].astype("Int32")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols

def balance_train(X_train: pd.DataFrame, y_train: np.ndarray, target_per_class: int = 300_000, seed: int = SEED):
    train = X_train.copy()
    train["__y"] = y_train
    counts = train["__y"].value_counts().to_dict()
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    print(f"[i] Train class counts before balance -> 0: {n0:,}, 1: {n1:,}")

    def under(df, k): return df.sample(n=min(len(df), k), random_state=seed)
    def over(df, k):
        if len(df) == 0: return df
        reps = math.ceil(k / len(df))
        out = pd.concat([df] * reps, ignore_index=True).sample(n=k, random_state=seed)
        return out

    g0 = train[train["__y"] == 0]
    g1 = train[train["__y"] == 1]

    if len(g0) <= len(g1):
        minority, majority = g0, g1
    else:
        minority, majority = g1, g0

    maj_bal = under(majority, target_per_class)
    min_bal = over(minority, target_per_class)

    balanced = pd.concat([maj_bal, min_bal], ignore_index=True).sample(frac=1.0, random_state=seed)
    yb = balanced["__y"].values
    Xb = balanced.drop(columns="__y")

    print(f"[i] Train class counts AFTER balance -> 0: {(yb==0).sum():,}, 1: {(yb==1).sum():,}")
    return Xb, yb

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
        class_weight=None
    )

    pipe = Pipeline([
        ("prep", pre),
        ("rf", model)
    ])
    return pipe

# ---------------- MAIN ----------------
def main():
    print("[i] Using datasets dir:", DATASETS_DIR)
    cols = load_columns(DATASETS_DIR)
    df_all = collect_sample(DATASETS_DIR, cols)

    label_col = None
    for cand in LABEL_CANDIDATES:
        if cand in df_all.columns:
            label_col = cand
            break
    if label_col is None:
        raise RuntimeError("No label column found.")

    X, y, num_cols, cat_cols = choose_features(df_all, label_col)
    print(f"[i] Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    print(f"[i] Split done: X_train={X_train.shape}, X_test={X_test.shape}")

    Xb, yb = balance_train(X_train, y_train)

    print("[*] Training model...")
    pipe = build_pipeline(Xb.select_dtypes(include=[np.number]).columns.tolist(),
                          [c for c in Xb.columns if c not in Xb.select_dtypes(include=[np.number]).columns])
    pipe.fit(Xb, yb)
    print("[OK] Model trained successfully")

    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_tpr": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "support": {"0": int((y_test==0).sum()), "1": int((y_test==1).sum())}
    }

    if proba is not None:
        metrics["auc"] = float(roc_auc_score(y_test, proba))
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        metrics["fpr"] = float(fp / (fp + tn) if (fp + tn) else 0)
        metrics["tnr_specificity"] = float(tn / (tn + fp) if (tn + fp) else 0)

    print(json.dumps(metrics, indent=2))

    bundle = {
        "pipeline": pipe,
        "label_col": label_col,
        "drop_cols": DROP_COLS,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "columns_from_names": cols
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[OK] Saved model to {MODEL_PATH} ({MODEL_PATH.stat().st_size/1e6:.2f} MB)")

    (REPORT_DIR / "metrics_combined.json").write_text(json.dumps(metrics, indent=2))
    print("[OK] Metrics saved to reports/metrics_combined.json")

if __name__ == "__main__":
    main()
