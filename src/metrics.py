"""Classification metrics in NumPy (no scikit-learn): accuracy, balanced accuracy, per-class P/R/F1,
macro-F1, confusion matrices, IRC-category accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.pcu import categories, class_to_category


def confusion(true: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """(k, k) counts; rows = true, columns = predicted."""
    return np.bincount(true * k + pred, minlength=k * k).reshape(k, k)


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support per class from a confusion matrix (0 where undefined)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": predicted}


def irc_prediction(cfg: dict[str, Any], softmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IRC category with the highest summed softmax over its classes, and that sum (the IRC score)."""
    c2k = class_to_category(cfg)
    summed = np.zeros((len(softmax), len(categories(cfg))))
    np.add.at(summed.T, c2k, softmax.T)
    return summed.argmax(1), summed.max(1)


def classification_report(cfg: dict[str, Any], true: np.ndarray, pred: np.ndarray,
                          softmax: np.ndarray, low_support: list[str]) -> dict[str, Any]:
    """Every classification number the report quotes for one arm on one split."""
    classes, k = cfg["dataset"]["classes"], len(cfg["dataset"]["classes"])
    cm = confusion(true, pred, k)
    pc = per_class(cm)
    present = pc["support"] > 0
    headline = present & ~np.isin(classes, low_support)
    c2k = class_to_category(cfg)
    cats = categories(cfg)
    irc_true = c2k[true]
    irc_pred, _ = irc_prediction(cfg, softmax)
    cm_irc = confusion(irc_true, irc_pred, len(cats))
    return {
        "n": int(len(true)),
        "accuracy": float((true == pred).mean()),
        "balanced_accuracy": float(pc["recall"][present].mean()),
        "macro_f1_headline": float(pc["f1"][headline].mean()),
        "macro_f1_all_present": float(pc["f1"][present].mean()),
        "macro_f1_classes_excluded": [c for c, h, p in zip(classes, headline, present) if p and not h],
        "irc_accuracy": float((irc_true == irc_pred).mean()),
        "irc_accuracy_from_class_argmax": float((irc_true == c2k[pred]).mean()),
        "per_class": {c: {"precision": round(float(pc["precision"][i]), 4), "recall": round(float(pc["recall"][i]), 4),
                          "f1": round(float(pc["f1"][i]), 4), "support": int(pc["support"][i]),
                          "predicted": int(pc["predicted"][i]), "low_support": c in low_support}
                      for i, c in enumerate(classes)},
        "confusion": cm.tolist(),
        "confusion_irc": cm_irc.tolist(),
        "irc_categories": cats,
    }
