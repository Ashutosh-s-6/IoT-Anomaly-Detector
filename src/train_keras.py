import json, os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .load_botiot import load_sample
from .preprocess import prepare_xy, split_train_test_scaled
from .metrics import evaluate_binary, plot_roc

def build_ann(input_dim: int):
    m = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def run_keras(target_per_class=5000, chunksize=100_000, max_files=None, epochs=10, batch_size=1024):
    df, _ = load_sample(
        target_per_class=target_per_class,
        chunksize=chunksize,
        max_files=max_files,
        allow_partial=True,
    )
    X, y = prepare_xy(df)
    print("Labels:", dict(zip(*np.unique(y, return_counts=True))))
    X_train, X_test, y_train, y_test, scaler = split_train_test_scaled(X, y)

    model = build_ann(X_train.shape[1])
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[es])

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics, cm = evaluate_binary(y_test, y_pred, y_proba)

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_keras.json", "w") as f:
        json.dump({"metrics": metrics, "cm": cm.tolist()}, f, indent=2)
    plot_roc(y_test, y_proba, "ROC (Keras ANN)", save_to="reports/roc_keras.png")

    print("== keras results ==")
    print("confusion matrix:\n", cm)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")

if __name__ == "__main__":
    run_keras()
