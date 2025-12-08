import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, mean_squared_error
)
import matplotlib.pyplot as plt
import os

def evaluate_binary(y_true, y_pred, y_proba=None) -> Tuple[Dict[str,float], np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)
    # ensure 2x2
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
    else:
        fpr = tpr = 0.0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (TPR)": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
        "TPR": tpr,
        "MSE": mean_squared_error(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["AUC-ROC"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["AUC-ROC"] = float("nan")

    return metrics, cm

def plot_roc(y_true, y_proba, title, save_to=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi=160)
    plt.show()
