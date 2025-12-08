# train_iot23_binary.py
import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# adjust paths if your folder is different
INPUT = r"C:/IoT-Anomaly-Detector/datasets/iot23_clean.parquet"
MODEL_OUT = r"C:/IoT-Anomaly-Detector/models/model_iot23.pkl"
FEATURES_OUT = r"C:/IoT-Anomaly-Detector/models/iot23_features.json"


def train():
    print("[+] Loading cleaned IoT-23 data...")
    df = pd.read_parquet(INPUT)
    print("[+] Full shape:", df.shape)
    print("[+] Class counts in full data:")
    print(df["attack"].value_counts())

    # ------------------------------
    # 1) BALANCE THE DATA
    # ------------------------------
    benign = df[df["attack"] == 0]
    attack = df[df["attack"] == 1]

    n_benign = len(benign)
    print(f"[+] Benign samples: {n_benign}")
    print(f"[+] Attack samples: {len(attack)}")

    # sample the same number of attacks as benign
    attack_sample = attack.sample(n_benign, random_state=42)

    df_bal = pd.concat([benign, attack_sample], axis=0)
    df_bal = df_bal.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("[+] After balancing:")
    print(df_bal["attack"].value_counts())
    print("[+] Balanced shape:", df_bal.shape)

    # ------------------------------
    # 2) SPLIT FEATURES / LABELS
    # ------------------------------
    y = df_bal["attack"].astype(int)
    X = df_bal.drop(columns=["attack"]).astype("float32")

    feature_names = list(X.columns)
    print("[+] Using features:", feature_names)

    # ------------------------------
    # 3) TRAIN / VAL SPLIT
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    print("[+] Train shape:", X_train.shape, "Val shape:", X_val.shape)

    # ------------------------------
    # 4) TRAIN RANDOM FOREST
    # ------------------------------
    print("[+] Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[OK] Model trained")

    # ------------------------------
    # 5) EVALUATE
    # ------------------------------
    print("[+] Evaluating on validation set...")
    preds = model.predict(X_val)
    print("[*] Confusion matrix:")
    print(confusion_matrix(y_val, preds))
    print("[*] Classification report:")
    print(classification_report(y_val, preds, digits=4))

    # ------------------------------
    # 6) SAVE MODEL + FEATURES
    # ------------------------------
    print("[+] Saving model and feature list...")
    joblib.dump(model, MODEL_OUT)

    with open(FEATURES_OUT, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)

    print(f"[✓] Model saved → {MODEL_OUT}")
    print(f"[✓] Feature list saved → {FEATURES_OUT}")


if __name__ == "__main__":
    train()
