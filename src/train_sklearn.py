import json, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .load_botiot import load_sample
from .preprocess import prepare_xy, split_train_test_scaled
from .metrics import evaluate_binary, plot_roc

def run_sklearn(target_per_class=5000, chunksize=100_000, max_files=None, model_type="rf"):
    # 1) get balanced/partial-balanced sample
    df, _ = load_sample(
        target_per_class=target_per_class,
        chunksize=chunksize,
        max_files=max_files,
        allow_partial=True,
    )

    # 2) raw X,y then split+scale without leakage
    X, y = prepare_xy(df)
    print("Labels:", dict(zip(*np.unique(y, return_counts=True))))
    X_train, X_test, y_train, y_test, scaler = split_train_test_scaled(X, y)

    # 3) model (RF doesn’t need scaling, but fine to keep for parity)
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] > 1:
            y_proba = proba[:, 1]

    # 4) metrics + outputs
    metrics, cm = evaluate_binary(y_test, y_pred, y_proba)
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_sklearn.json", "w") as f:
        json.dump({"metrics": metrics, "cm": cm.tolist()}, f, indent=2)
    if y_proba is not None:
        plot_roc(y_test, y_proba, "ROC (RF)", save_to="reports/roc_sklearn.png")
    else:
        print("⚠️ Skipping ROC (single-class probabilities).")

    print("== sklearn results ==")
    print("confusion matrix:\n", cm)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")

if __name__ == "__main__":
    run_sklearn()
