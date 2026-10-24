"""Shared evaluation utilities for all three tiers.

Every tier MUST use these functions so that metrics are computed identically.

Convention:
  - label 1 = anomaly (defect), 0 = normal
  - higher score = more anomalous
  - prediction rule: anomaly  <=>  score >= tau

Usage:
    from evaluate import evaluate, per_type_recall, find_best_f1_threshold

    tau = find_best_f1_threshold(y_val, scores_val)          # choose tau on VAL only
    metrics = evaluate(y_test, scores_test, tau)               # evaluate on TEST once
    recalls = per_type_recall(failure_types_test, scores_test, tau)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict(scores, tau):
    """Binary prediction: 1 (anomaly) if score >= tau."""
    return (np.asarray(scores) >= tau).astype(int)


def evaluate(y_true, scores, tau):
    """Threshold-free and threshold-based metrics.

    Returns a dict:
      pr_auc   : average precision (primary metric; baseline = anomaly prevalence)
      roc_auc  : ROC-AUC (0.5 = random)
      f1, precision, recall : at the given tau
      fpr      : false alarm rate = FP / (FP + TN)
      accuracy : reported only to show the accuracy pitfall under imbalance
      tp, fp, fn, tn : confusion matrix counts
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = predict(scores, tau)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "accuracy": accuracy_score(y_true, y_pred),
        "tau": tau,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def per_type_recall(failure_types, scores, tau):
    """Detection rate per failureType at tau.

    For defect types this is recall (share detected).
    For 'none' it is the false alarm rate (share of normals flagged).
    """
    df = pd.DataFrame({"failureType": np.asarray(failure_types), "pred": predict(scores, tau)})
    out = df.groupby("failureType")["pred"].agg(n="size", detected="sum")
    out["rate"] = out["detected"] / out["n"]
    return out


def find_best_f1_threshold(y_val, scores_val):
    """Threshold that maximizes F1 on the validation set. Never call this on test data."""
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    # precision/recall have one more element than thresholds; drop the last point
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])
