# backend/app.py
import os, io, glob, json, warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_squared_error
)

# silence annoying sklearn feature-name warning
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but .* was fitted without feature names"
)

app = Flask(__name__)
CORS(app, supports_credentials=True)

HERE = Path(__file__).parent
FRONTEND_DIR = (HERE / ".." / "frontend").resolve()
ASSETS_DIR = (HERE / ".." / "assets").resolve()
DEVICES_MAP_PATH = (HERE / ".." / "datasets" / "devices_map.csv").resolve()

# ---------------- Load multiple models (Bot-IoT + IoT-23) ----------------
def load_bot_iot_model():
    """
    Balanced Bot-IoT model (BotIotmodel.pkl or model.pkl fallback) from backend/.
    """
    # prefer new balanced model; fall back to old name if needed
    for name in ("BotIotmodel.pkl", "model.pkl"):
        path = HERE / name
        if path.exists():
            break
    else:
        return None

    bundle = joblib.load(path)
    pipe = bundle["pipeline"]

    label_col = bundle.get("label_col", "attack")
    drop_cols = bundle.get("drop_cols", [])
    cat_cols  = list(bundle.get("categorical_cols", []))
    num_cols  = list(bundle.get("numeric_cols", []))
    cols_from_names = bundle.get("columns_from_names", None)

    feat_cols = bundle.get("feature_cols")
    if feat_cols is None:
        feat_cols = num_cols + cat_cols

    return {
        "key": "bot-iot",
        "name": "Bot-IoT",
        "pipe": pipe,
        "label_col": label_col,
        "drop_cols": drop_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_cols": list(feat_cols),
        "columns_from_names": cols_from_names,
    }


def load_iot23_model():
    """
    IoT-23 model from project-root/models/model_iot23.pkl
    + feature order from models/iot23_features.json.
    """
    models_dir = (HERE / ".." / "models").resolve()
    model_path = models_dir / "model_iot23.pkl"
    feats_path = models_dir / "iot23_features.json"

    if not model_path.exists():
        return None

    bundle = joblib.load(model_path)

    # model
    if isinstance(bundle, dict):
        pipe = bundle.get("pipeline") or bundle.get("model")
        feat_cols = bundle.get("feature_cols")
    else:
        pipe = bundle
        feat_cols = getattr(pipe, "feature_names_in_", None)

    # Convert ndarray → list
    if isinstance(feat_cols, np.ndarray):
        feat_cols = feat_cols.tolist()

    # If still empty → load from JSON feature list
    if (feat_cols is None or len(feat_cols) == 0) and feats_path.exists():
        try:
            with open(feats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                feat_cols = data.get("features", [])
        except Exception:
            feat_cols = []

    feat_cols = list(feat_cols) if feat_cols else []

    return {
        "key": "iot23",
        "name": "IoT-23",
        "pipe": pipe,
        "label_col": "attack",
        "drop_cols": [],
        "cat_cols": [],
        "num_cols": feat_cols,
        "feature_cols": feat_cols,
        "columns_from_names": None,  # infer CSV header
    }



MODELS: dict[str, dict] = {}

for loader in (load_bot_iot_model, load_iot23_model):
    cfg = loader()
    if cfg is not None:
        MODELS[cfg["key"]] = cfg

if not MODELS:
    raise RuntimeError("No models found (neither Bot-IoT nor IoT-23).")

DEFAULT_MODEL_KEY = next(iter(MODELS))  # first one as default


def get_model_config(model_key: str | None) -> dict:
    """Return model config; fall back to default if key invalid/missing."""
    if not model_key:
        return MODELS[DEFAULT_MODEL_KEY]
    return MODELS.get(model_key, MODELS[DEFAULT_MODEL_KEY])

# ---------------- JSON sanitizer ----------------
def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj

# ---------------- Vendor & Device Map ----------------
def load_vendor_db():
    sample = {
        "00:1A:2B": "Cisco",
        "F4:92:BF": "Huawei",
        "44:65:0D": "Samsung",
        "D8:9E:F3": "Xiaomi",
        "BC:92:6B": "TP-Link",
    }
    path = ASSETS_DIR / "oui.json"
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k.strip().upper(): v for k, v in data.items()}
    except Exception:
        pass
    return sample

def mac_prefix(mac: str) -> str | None:
    if not mac or not isinstance(mac, str):
        return None
    parts = mac.strip().upper().replace("-", ":").split(":")
    if len(parts) < 3:
        return None
    return ":".join(parts[:3])

VENDOR_DB = load_vendor_db()

def vendor_from_mac(mac: str) -> str | None:
    pref = mac_prefix(mac)
    if not pref:
        return None
    return VENDOR_DB.get(pref)

def load_devices_map():
    mp = {}
    try:
        if DEVICES_MAP_PATH.exists():
            dfm = pd.read_csv(DEVICES_MAP_PATH)
            for _, r in dfm.iterrows():
                ip = str(r.get("ip", "")).strip()
                if not ip:
                    continue
                mp[ip] = {
                    "name": str(r.get("name", "")).strip() or None,
                    "type": str(r.get("type", "")).strip() or None,
                }
    except Exception:
        pass
    return mp

DEVICES_MAP = load_devices_map()

# ---------------- Helpers ----------------
def detect_header_bytes(raw_bytes, columns):
    if columns is None:
        return False
    try:
        first_line = raw_bytes.decode("utf-8", errors="ignore").splitlines()[0]
        return (
            [x.strip().lower() for x in first_line.split(",")]
            == [x.strip().lower() for x in columns]
        )
    except Exception:
        return False

def read_uploaded_csv(file_storage, columns=None):
    raw = file_storage.read()
    file_storage.stream.seek(0)
    skiprows = 1 if detect_header_bytes(raw, columns) else 0
    df = pd.read_csv(
        io.BytesIO(raw),
        names=columns if columns is not None else None,
        header=None if columns is not None else "infer",
        skiprows=skiprows,
        low_memory=False,
    )
    return df

def normalize_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0, "attack": 1, "attacks": 1, "anomaly": 1, "malicious": 1,
        "true": 1, "yes": 1, "normal": 0, "benign": 0, "false": 0, "no": 0
    }
    out = s.map(mapping)
    num = pd.to_numeric(series, errors="coerce")
    out = out.where(~out.isna(), num)
    keep = ~out.isna()
    return out.loc[keep].astype(int), keep

def prepare_X(df: pd.DataFrame, cfg: dict):
    """
    Build the feature matrix for the selected model.

    - For IoT-23: rebuild each feature column explicitly as numeric
      (any strings like 'S0', 'SF', 'duration', etc. become NaN -> 0).
    - For Bot-IoT: keep existing behaviour (since that pipeline handles
      categorical features like proto/service).
    """
    label_col = cfg["label_col"]
    drop_cols = cfg["drop_cols"]
    feat_cols = cfg["feature_cols"]

    # ---------------- IoT-23: fully numeric, very strict ----------------
    if cfg.get("key") == "iot23":
        X_cols = {}

        # we ONLY use the known feature columns, in the right order
        for col in feat_cols:
            if col in df.columns:
                s = df[col]
            else:
                # missing column -> fill with zeros
                s = 0

            # force numeric, anything non-numeric becomes NaN then 0
            s_num = pd.to_numeric(s, errors="coerce").fillna(0)
            X_cols[col] = s_num

        X = pd.DataFrame(X_cols)
        return X

    # ---------------- Bot-IoT: previous logic (categoricals etc.) ------

    # Drop label + any configured extra columns
    cols_to_drop = [label_col] + [c for c in drop_cols if c in df.columns]
    X = df.drop(cols_to_drop, axis=1, errors="ignore").copy()

    # Ensure all expected feature columns exist and are ordered correctly
    if feat_cols:
        missing = [c for c in feat_cols if c not in X.columns]
        for c in missing:
            X[c] = 0
        X = X[feat_cols]

    return X



def find_best_threshold(y: np.ndarray, scores: np.ndarray, metric: str = "f1") -> tuple[float, dict]:
    """
    Scan thresholds in [0.01, 0.99] to find best one according to
    - metric="f1" (default): maximize F1
    - metric="youden": maximize TPR - FPR
    """
    # subsample for speed if huge
    if len(y) > 200_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), size=200_000, replace=False)
        y_s = y[idx]
        s_s = scores[idx]
    else:
        y_s, s_s = y, scores

    candidates = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_val = -1.0
    best_stats: dict = {}

    for t in candidates:
        preds = (s_s >= t).astype(int)
        prec = precision_score(y_s, preds, zero_division=0)
        rec  = recall_score(y_s, preds, zero_division=0)
        f1   = f1_score(y_s, preds, zero_division=0)
        cm   = confusion_matrix(y_s, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        if metric == "youden":
            score_val = rec - fpr
        else:  # default F1
            score_val = f1

        if score_val > best_val:
            best_val = score_val
            best_thr = float(t)
            best_stats = {
                "precision": float(prec),
                "recall_tpr": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
            }

    best_stats["metric"] = metric
    best_stats["threshold"] = best_thr
    return best_thr, best_stats

# ---------------- Static (dashboard) ----------------
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.get("/<path:path>")
def static_proxy(path):
    return send_from_directory(FRONTEND_DIR, path)

# ---------------- API ----------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models": [
            {"key": k, "name": cfg["name"]}
            for k, cfg in MODELS.items()
        ],
        "default_model": DEFAULT_MODEL_KEY,
    })

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file missing"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400

        # model selection
        model_key = request.form.get("model") or request.args.get("model")
        cfg = get_model_config(model_key)
        pipe = cfg["pipe"]
        cols_from_names = cfg["columns_from_names"]
        label_col = cfg["label_col"]

        df = read_uploaded_csv(f, columns=cols_from_names)

        # label column candidate
        label_candidate = None
        for cand in [label_col, "attack", "label"]:
            if cand in df.columns:
                label_candidate = cand
                break

        X = prepare_X(df, cfg)

        # scores
        scores = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        elif hasattr(pipe, "decision_function"):
            decision = pipe.decision_function(X)
            scores = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)

        # base threshold from user slider (default 0.5)
        thr_param = request.form.get("threshold") or request.args.get("threshold")
        threshold = float(thr_param) if thr_param is not None else 0.5

        auto_flag = request.form.get("auto_threshold") or request.args.get("auto_threshold")
        auto_metric = request.form.get("auto_metric") or request.args.get("auto_metric") or "youden"

        auto_info = None

        # if we have labels + scores and auto-threshold flag, override threshold
        y = None
        keep = None
        if label_candidate is not None:
            y_raw = df[label_candidate]
            y, keep = normalize_label(y_raw)

            if auto_flag and scores is not None:
                scores_lab = scores[keep.values]
                threshold, auto_info = find_best_threshold(y.values, scores_lab, metric=auto_metric)

        # predictions using final threshold
        if scores is not None:
            preds = (scores >= threshold).astype(int)
        else:
            preds = pipe.predict(X).astype(int)

        total = int(len(X))
        anomalies = int((preds == 1).sum())
        normals = int(total - anomalies)

        metrics = None
        cm_json = None
        roc_json = None

        # compute metrics only if we have true labels
        if y is not None:
            preds_lab = preds[keep.values]
            scores_lab = scores[keep.values] if scores is not None else None

            acc = accuracy_score(y, preds_lab)
            prec = precision_score(y, preds_lab, zero_division=0)
            rec  = recall_score(y, preds_lab, zero_division=0)
            f1v  = f1_score(y, preds_lab, zero_division=0)
            mse  = mean_squared_error(y, preds_lab if scores_lab is None else (scores_lab >= threshold).astype(int))
            cm   = confusion_matrix(y, preds_lab, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0

            metrics = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall_tpr": float(rec),
                "f1": float(f1v),
                "fpr": float(fpr),
                "tnr_specificity": float(tnr),
                "mse": float(mse),
                "support": {"0": int((y == 0).sum()), "1": int((y == 1).sum())}
            }

            if scores_lab is not None:
                try:
                    auc = roc_auc_score(y, scores_lab)
                except Exception:
                    auc = float("nan")
                metrics["auc"] = float(auc)

                fpr_vals, tpr_vals, thr_vals = roc_curve(y, scores_lab)
                thr_vals = np.where(np.isfinite(thr_vals), thr_vals, np.nan)
                roc_json = {
                    "fpr": fpr_vals.tolist(),
                    "tpr": tpr_vals.tolist(),
                    "thresholds": thr_vals.tolist(),
                    "auc": float(metrics["auc"]),


                }

            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = (cm / row_sums).round(4).tolist()
            cm_json = {"labels": ["0", "1"], "matrix": cm.tolist(), "matrix_pct": cm_pct}

        out = {
            "model": cfg["key"],
            "summary": {
                "total_records": total,
                "anomalies_detected": anomalies,
                "normal_records": normals,
                "threshold": threshold,
                "has_proba": bool(scores is not None),
            },
            "auto_threshold": auto_info,
            "metrics": metrics,
            "confusion": cm_json,
            "roc": roc_json,
            "preview": {
                "predictions": preds[:20].tolist(),
                "scores": (np.round(scores[:20], 6).tolist() if scores is not None else None),
            },
        }

        return jsonify(json_safe(out)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/devices")
def devices():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file missing"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "empty filename"}), 400

        model_key = request.form.get("model") or request.args.get("model")
        cfg = get_model_config(model_key)
        pipe = cfg["pipe"]
        cols_from_names = cfg["columns_from_names"]

        df = read_uploaded_csv(f, columns=cols_from_names)

        mode = (request.form.get("group_by") or request.args.get("group_by") or "auto").lower()
        if mode not in {"auto", "src", "dst", "both"}:
            mode = "auto"

        has_s = "saddr" in df.columns
        has_d = "daddr" in df.columns

        X = prepare_X(df, cfg)

        scores = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        elif hasattr(pipe, "decision_function"):
            decision = pipe.decision_function(X)
            scores = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)

        thr = request.form.get("threshold") or request.args.get("threshold")
        threshold = float(thr) if thr is not None else 0.5
        preds = (scores >= threshold).astype(int) if scores is not None else pipe.predict(X).astype(int)

        df["_pred"] = preds
        if scores is not None:
            df["_score"] = scores

        attack_col = None
        for cand in ["category", "subcategory"]:
            if cand in df.columns:
                attack_col = cand
                break

        frames = []

        include_src = (mode in {"src", "both"}) or (mode == "auto" and has_s)
        include_dst = (mode in {"dst", "both"}) or (mode == "auto" and has_d)

        if include_src and has_s:
            fs = pd.DataFrame({
                "dev": df["saddr"],
                "_pred": df["_pred"],
                "pkts": pd.to_numeric(df.get("pkts", 0), errors="coerce"),
                "bytes": pd.to_numeric(df.get("bytes", 0), errors="coerce"),
                "ltime": df.get("ltime"),
                "smac": df.get("smac"),
                "dmac": df.get("dmac"),
                "attack": df.get(attack_col) if attack_col else None,
            })
            fs["role"] = "src"
            frames.append(fs)

        if include_dst and has_d:
            fd = pd.DataFrame({
                "dev": df["daddr"],
                "_pred": df["_pred"],
                "pkts": pd.to_numeric(df.get("pkts", 0), errors="coerce"),
                "bytes": pd.to_numeric(df.get("bytes", 0), errors="coerce"),
                "ltime": df.get("ltime"),
                "smac": df.get("smac"),
                "dmac": df.get("dmac"),
                "attack": df.get(attack_col) if attack_col else None,
            })
            fd["role"] = "dst"
            frames.append(fd)

        if not frames:
            return jsonify({"error": "No saddr/daddr columns to group"}), 400

        df_long = pd.concat(frames, ignore_index=True)

        df_long["dev"] = df_long["dev"].astype(str).str.strip()
        df_long = df_long.replace({"dev": {"": np.nan, "nan": np.nan}}).dropna(subset=["dev"])

        df_long["pkts"]  = pd.to_numeric(df_long["pkts"],  errors="coerce").fillna(0)
        df_long["bytes"] = pd.to_numeric(df_long["bytes"], errors="coerce").fillna(0)

        suniq = df["saddr"].nunique(dropna=True) if has_s else 0
        duniq = df["daddr"].nunique(dropna=True) if has_d else 0
        uuniq = pd.concat([
            df.get("saddr", pd.Series(dtype=object)),
            df.get("daddr", pd.Series(dtype=object))
        ], ignore_index=True).dropna().nunique()

        devices = []
        grouped = df_long.groupby("dev", dropna=False)

        for dev, g in grouped:
            total = int(len(g))
            anomalies = int((g["_pred"] == 1).sum())
            anomaly_rate = anomalies / total if total else 0.0
            status = "RED" if anomalies > 0 else "GREEN"

            top_attack = None
            if attack_col:
                try:
                    aa = g.loc[g["_pred"] == 1, "attack"].astype(str).str.strip()
                    aa = aa.replace("", np.nan).dropna()
                    if len(aa):
                        top_attack = aa.value_counts().idxmax()
                    else:
                        bb = g["attack"].astype(str).str.strip().replace("", np.nan).dropna()
                        if len(bb):
                            top_attack = bb.value_counts().idxmax()
                except Exception:
                    top_attack = None

            pkts = int(g["pkts"].sum())
            bytes_ = int(g["bytes"].sum())
            last_seen = g["ltime"].max() if "ltime" in g.columns else None

            smac = g["smac"].dropna().astype(str)
            dmac = g["dmac"].dropna().astype(str)
            vendor = None
            for mac in list(smac[:3]) + list(dmac[:3]):
                vendor = vendor_from_mac(mac)
                if vendor:
                    break

            dev_ip = str(dev) if dev is not None else ""
            mapped = DEVICES_MAP.get(dev_ip, {})
            name = mapped.get("name")
            dtype = mapped.get("type")

            devices.append({
                "device": dev_ip,
                "name": name,
                "dtype": dtype,
                "vendor": vendor,
                "status": status,
                "total": total,
                "anomalies": anomalies,
                "anomaly_rate": anomaly_rate,
                "top_attack": top_attack,
                "totals": {"pkts": pkts, "bytes": bytes_},
                "last_seen": last_seen,
                "smac": smac.iloc[0] if len(smac) else None,
                "dmac": dmac.iloc[0] if len(dmac) else None,
            })

        out = {
            "model": cfg["key"],
            "dev_col": ("both" if (include_src and include_dst) else ("saddr" if include_src else "daddr")),
            "threshold": threshold,
            "devices": devices,
            "dev_stats": {"unique_saddr": int(suniq), "unique_daddr": int(duniq), "unique_union": int(uuniq)}
        }
        return jsonify(json_safe(out)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
