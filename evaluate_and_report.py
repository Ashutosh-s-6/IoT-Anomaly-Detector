# evaluate_and_report.py
import os, glob, json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, mean_squared_error
)
import matplotlib.pyplot as plt

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True, parents=True)

def detect_header(csv_path, columns):
    with open(csv_path, "r") as f:
        head = f.readline().strip().split(",")
    return [h.strip().lower() for h in head] == [c.strip().lower() for c in columns]

def load_holdout_batch(datasets_dir, columns, target_col="attack",
                       total_rows=200_000, pos_frac=0.5, skip_first_n_files=5):
    """
    Builds a balanced holdout by sampling positives/negatives across files.
    pos_frac=0.5 -> 50/50 positive/negative by default.
    """
    files = sorted(glob.glob(os.path.join(datasets_dir, "data_*.csv")))
    files = files[skip_first_n_files:] or files  # if not enough files, reuse

    need_pos = int(total_rows * pos_frac)
    need_neg = total_rows - need_pos
    pos_parts, neg_parts = [], []

    for p in files:
        has_header = detect_header(p, columns)
        df = pd.read_csv(
            p, names=columns, header=None,
            skiprows=1 if has_header else 0, low_memory=False
        )

        if target_col not in df.columns:
            raise RuntimeError(f"'{target_col}' not in {p}")

        # normalize label quickly (binary)
        y = df[target_col].astype(str).str.strip().str.lower().map(
            {"1":1,"0":0,"attack":1,"normal":0,"benign":0,"true":1,"false":0,"yes":1,"no":0}
        )
        keep = ~y.isna()
        if not keep.any():
            continue
        df = df.loc[keep]
        y  = y.loc[keep].astype(int)

        pos = df[y == 1]
        neg = df[y == 0]

        if len(pos) > 0 and need_pos > 0:
            take = min(len(pos), need_pos)
            pos_parts.append(pos.sample(take, random_state=42))
            need_pos -= take

        if len(neg) > 0 and need_neg > 0:
            take = min(len(neg), need_neg)
            neg_parts.append(neg.sample(take, random_state=42))
            need_neg -= take

        if need_pos <= 0 and need_neg <= 0:
            break

    if not pos_parts and not neg_parts:
        raise RuntimeError("Could not assemble a holdout batch. Check label mapping and files.")

    df_out = pd.concat(pos_parts + neg_parts, ignore_index=True)
    # shuffle
    df_out = df_out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df_out

def main():
    # 1) Load model bundle
    bundle = joblib.load("backend/model.pkl")
    pipe = bundle["pipeline"]
    label_col = bundle["label_col"]
    drop_cols = bundle["drop_cols"]
    columns_from_names = bundle["columns_from_names"]

    # 2) Dataset dir + columns
    datasets_dir = "datasets" if os.path.isfile("datasets/data_names.csv") else "../datasets"
    cols = pd.read_csv(os.path.join(datasets_dir, "data_names.csv"), header=None).iloc[0].tolist()

    # 3) Balanced holdout (300k rows, 50/50 split)
    df = load_holdout_batch(
        datasets_dir, cols, target_col=label_col,
        total_rows=300_000, pos_frac=0.5, skip_first_n_files=5
    )

    # Prepare X,y
    if label_col not in df.columns:
        raise RuntimeError(f"Label column '{label_col}' not found in holdout data.")

    y = df[label_col].astype(str).str.strip().str.lower().map(
        {"1":1,"0":0,"attack":1,"normal":0,"benign":0,"true":1,"false":0,"yes":1,"no":0}
    )
    keep = ~y.isna()
    df = df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)

    X = df.drop([label_col] + [c for c in drop_cols if c in df.columns], axis=1, errors="ignore")

    # 4) Predict
    y_pred = pipe.predict(X)

    # Probabilities for ROC/AUC
    y_proba = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]

    # 5) Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)  # TPR
    f1  = f1_score(y, y_pred, zero_division=0)
    mse = mean_squared_error(y, y_pred)
    cm  = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0  # Specificity

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall_tpr": rec,
        "f1": f1,
        "fpr": fpr,
        "tnr_specificity": tnr,
        "mse": mse,
        "support": {"0": int((y==0).sum()), "1": int((y==1).sum())}
    }

    if y_proba is not None:
        try:
            auc = roc_auc_score(y, y_proba)
        except Exception:
            auc = float("nan")
        metrics["auc"] = auc

    # 6) Save metrics + confusion (JSON + CSV)
    (REPORT_DIR / "metrics_rf.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "metrics_rf.csv", index=False)

    (REPORT_DIR / "confusion_rf.json").write_text(json.dumps({
        "labels": ["0","1"],
        "matrix": cm.tolist(),
        "matrix_pct": (cm / cm.sum()).round(4).tolist()
    }, indent=2))

    print("[OK] Metrics saved to reports/metrics_rf.json and metrics_rf.csv")
    print(json.dumps(metrics, indent=2))

    # 7) Plots — Confusion Matrix
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (RF)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0","1"])
    plt.yticks(tick_marks, ["0","1"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_rf.png", dpi=150)
    plt.close()

    # ROC
    if y_proba is not None:
        fpr_vals, tpr_vals, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr_vals, tpr_vals, label=f"AUC={metrics.get('auc', float('nan')):.3f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR")
        plt.ylabel("TPR (Recall)")
        plt.title("ROC Curve (RF)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "roc_rf.png", dpi=150)
        plt.close()
        (REPORT_DIR / "roc_rf.json").write_text(json.dumps({
            "fpr": fpr_vals.tolist(), "tpr": tpr_vals.tolist(), "auc": metrics.get("auc", None)
        }, indent=2))

if __name__ == "__main__":
    main()
