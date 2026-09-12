from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ["FULL_PPE", "PARTIAL_PPE", "NO_PPE"],
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class metrics including safety-critical false negative rates.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    num_classes = len(class_names)
    labels = list(range(num_classes))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_p = float(precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0))

    per_class_p = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0).tolist()
    per_class_r = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0).tolist()

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # Safety Specific Metrics: Unsafe Class Recalls & Critical False Negatives
    # Target Classes: 0: FULL_PPE, 1: PARTIAL_PPE, 2: NO_PPE
    # Critical Safety Hazard 1: Worker has NO_PPE (2) but model predicts FULL_PPE (0)
    no_ppe_total = int(np.sum(y_true_arr == 2))
    no_ppe_predicted_full = int(np.sum((y_true_arr == 2) & (y_pred_arr == 0)))
    no_ppe_critical_fn_rate = (no_ppe_predicted_full / float(no_ppe_total)) if no_ppe_total > 0 else 0.0

    # Critical Safety Hazard 2: Worker has PARTIAL_PPE (1) but model predicts FULL_PPE (0)
    partial_ppe_total = int(np.sum(y_true_arr == 1))
    partial_predicted_full = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    partial_ppe_fn_rate = (partial_predicted_full / float(partial_ppe_total)) if partial_ppe_total > 0 else 0.0

    unsafe_recall_min = min(per_class_r[1], per_class_r[2])
    unsafe_recall_mean = (per_class_r[1] + per_class_r[2]) / 2.0

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            cls_name: {
                "precision": round(per_class_p[i], 4),
                "recall": round(per_class_r[i], 4),
                "f1": round(per_class_f1[i], 4),
                "support": int(np.sum(y_true_arr == i)),
            }
            for i, cls_name in enumerate(class_names)
        },
        "safety_audit": {
            "unsafe_recall_min": round(unsafe_recall_min, 4),
            "unsafe_recall_mean": round(unsafe_recall_mean, 4),
            "no_ppe_critical_fn_count": no_ppe_predicted_full,
            "no_ppe_critical_fn_rate": round(no_ppe_critical_fn_rate, 4),
            "partial_ppe_fn_count": partial_predicted_full,
            "partial_ppe_fn_rate": round(partial_ppe_fn_rate, 4),
        },
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": [[round(float(v), 4) for v in row] for row in cm_norm],
    }

    return metrics
