# train_and_save.py
import os, glob, csv
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ---------- CONFIG ----------
DATASET_DIRS = ["datasets", "../datasets", "../../datasets"]  # common locations to try
DATA_NAMES = "data_names.csv"
DATA_GLOB = "data_*.csv"

N_FILES = 5            # how many CSV parts to read (start small; scale later)
N_ROWS_PER_FILE = 50_000
LABEL_COLS = ["attack", "label"]  # first one found will be used

# columns to drop if present (IDs, device-specific, timestamps)
DROP_COLS = [
    "pkSeqID","stime","ltime","seq","saddr","daddr","sport","dport",
    "smac","dmac","soui","doui","flow_id","timestamp"
]

# likely categorical columns to one-hot encode (if present)
CATEGORICAL_GUESS = ["proto","flgs","state"]

MODEL_OUT = os.path.join("backend", "model.pkl")
# ----------------------------


def find_dataset_dir():
    for d in DATASET_DIRS:
        if os.path.isfile(os.path.join(d, DATA_NAMES)):
            return d
    raise FileNotFoundError(
        f"Could not find {DATA_NAMES} in any of: {DATASET_DIRS}. "
        "Run from project root or adjust DATASET_DIRS."
    )


def load_columns(datasets_dir: str):
    cols = pd.read_csv(os.path.join(datasets_dir, DATA_NAMES), header=None).iloc[0].tolist()
    return [c.strip() for c in cols]


def load_sample(datasets_dir: str, columns):
    paths = sorted(glob.glob(os.path.join(datasets_dir, DATA_GLOB)))
    if not paths:
        raise FileNotFoundError(f"No CSV parts found matching {DATA_GLOB} in {datasets_dir}")

    # Detect if part files contain a header row by peeking at first line
    with open(paths[0], "r", newline="") as f:
        first_line = f.readline().strip()
    has_header_in_parts = [x.strip().lower() for x in first_line.split(",")] == \
                          [x.strip().lower() for x in columns]

    dfs = []
    for p in paths[:N_FILES]:
        part = pd.read_csv(
            p,
            names=columns,          # enforce schema
            header=None,            # never treat first row as header
            skiprows=1 if has_header_in_parts else 0,  # drop header line if present
            low_memory=False
        )
        dfs.append(part)
    df = pd.concat(dfs, ignore_index=True)
    return df


def pick_label(df: pd.DataFrame) -> str:
    for c in LABEL_COLS:
        if c in df.columns:
            return c
    raise KeyError(f"None of {LABEL_COLS} found in columns: {df.columns.tolist()}")


def normalize_label(series: pd.Series) -> pd.Series:
    """
    Map various encodings to {0,1}. Handles header-leak & text labels.
    """
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0, "1": 1,
        "attack": 1, "attacks": 1, "anomaly": 1, "malicious": 1, "true": 1, "yes": 1,
        "normal": 0, "benign": 0, "false": 0, "no": 0
    }
    mapped = s.map(mapping)

    # Try numeric fallback for anything unmapped
    numeric = pd.to_numeric(series, errors="coerce")
    out = mapped.astype("float64")
    out = out.where(~out.isna(), numeric)

    # Drop rows that are still NaN
    keep_mask = ~out.isna()
    if (~keep_mask).any():
        # silently drop bad rows (usually header leftovers)
        out = out[keep_mask]

    return out.astype(int)


def clean_features(df: pd.DataFrame, label_col: str):
    """
    Drop non-ML columns; coerce numerics; leave likely categoricals for OHE.
    """
    drop_actual = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(drop_actual + [label_col], axis=1, errors="ignore").copy()

    # Guess categoricals
    cat_guess = [c for c in CATEGORICAL_GUESS if c in X.columns]

    # Coerce non-categorical columns to numeric
    for c in X.columns:
        if c not in cat_guess:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Replace inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that became entirely NaN
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # Fill remaining NaNs
    X = X.fillna(0)

    return X, cat_guess, drop_actual


def build_pipeline(feature_df: pd.DataFrame, cat_cols):
    # Numeric = everything not in cat_cols
    num_cols = [c for c in feature_df.columns if c not in cat_cols]

    # sklearn version–compatible OneHotEncoder (handles both new/old signatures)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn < 1.2

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, [c for c in cat_cols if c in feature_df.columns]),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe, num_cols



def main():
    os.makedirs("backend", exist_ok=True)

    datasets_dir = find_dataset_dir()
    print(f"[i] Using datasets dir: {datasets_dir}")

    columns = load_columns(datasets_dir)
    print(f"[i] Loaded {len(columns)} columns from data_names.csv")

    df = load_sample(datasets_dir, columns)
    print(f"[i] Loaded sample: {df.shape}")

    # choose & normalize label
    label_col = pick_label(df)
    y = normalize_label(df[label_col])

    # align df to kept rows (normalize_label may drop header-leak rows)
    df = df.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    print(f"[i] After label normalization: {df.shape} rows")

    # build features
    X, cat_cols, drop_actual = clean_features(df, label_col)
    print(f"[i] Features: {X.shape[1]} (numeric + {len(cat_cols)} categorical) | Dropped: {drop_actual}")

    # build pipeline
    pipe, num_cols = build_pipeline(X, cat_cols)

    # split + train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # quick sanity metric
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[OK] Trained RF pipeline. Accuracy on holdout: {acc:.4f}")

    # save bundle (pipeline + metadata)
    bundle = {
        "pipeline": pipe,
        "label_col": label_col,
        "drop_cols": drop_actual,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "columns_from_names": columns,
    }
    joblib.dump(bundle, MODEL_OUT)
    size = os.path.getsize(MODEL_OUT)
    print(f"[OK] Saved model to {MODEL_OUT} ({size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
