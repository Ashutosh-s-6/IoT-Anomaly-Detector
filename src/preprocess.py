import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# columns that are identifiers / strings we don't want as features
DROP_COLS_DEFAULT = ["src_ip", "dst_ip", "sport", "dport", "state", "timestamp", "flow_id"]

def prepare_xy(df: pd.DataFrame, label_col: str = "label"):
    """
    Make raw features X (unscaled) and labels y from Bot-IoT frames.
    - prefers canonical 'label' column (0/1)
    - coerces all features to numeric, drops non-numeric
    """
    preferred = ["label", "Label", "attack", "category", "class"]
    if label_col not in df.columns:
        for c in preferred:
            if c in df.columns:
                label_col = c
                break
        else:
            raise ValueError("❌ No label column found in dataset!")

    if "label" in df.columns:
        label_col = "label"

    print(f"✅ Using label column: {label_col}")

    y_raw = df[label_col]
    if y_raw.dtype == "object":
        y = (
            y_raw.astype(str).str.lower().map({
                "normal": 0, "benign": 0,
                "attack": 1, "anomaly": 1,
                "ddos": 1, "dos": 1, "theft": 1,
                "scan": 1, "keylogging": 1, "data_exfiltration": 1
            }).fillna(1).astype(int)
        )
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(1).astype(int).clip(0, 1)

    drop_cols = [c for c in (DROP_COLS_DEFAULT + [label_col]) if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    if X.shape[1] == 0:
        raise ValueError("❌ No numeric features survived. Dataset may not be parsed correctly!")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"✅ Features used: {X.shape[1]} numeric columns")
    return X, y

def split_train_test_scaled(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, seed: int = 42):
    """
    Stratified split, then fit scaler on X_train only, transform both.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler
