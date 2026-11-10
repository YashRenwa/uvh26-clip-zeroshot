"""Passenger Car Units, ported from Assignment 1 (src/counts.py::pcu_factors, error_summary).

IRC:106-1990 Table 1 factors at 5% and >=10% share of the traffic stream, linearly interpolated by each
IRC category's share of A1's train ST boxes; the 14 class factors are asserted equal to A1's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# A1's printed factors (A1 report/outputs/outputs.md, PCU factor table).
A1_CLASS_FACTOR = {
    "Hatchback": 1.0, "Sedan": 1.0, "SUV": 1.0, "MUV": 1.0, "Van": 1.0,
    "Bus": 3.276, "Truck": 3.276, "Mini-bus": 3.276,
    "Tempo-traveller": 1.5401, "LCV": 1.5401,
    "Three-wheeler": 2.0, "Two-wheeler": 0.75, "Bicycle": 0.4, "Others": 4.0,
}


def pcu_factors(cfg: dict[str, Any]) -> dict[str, Any]:
    """Factor per UVH-26 class from the IRC categories' shares of A1's train traffic (ST labels)."""
    p, classes = cfg["pcu"], cfg["dataset"]["classes"]
    totals = {c: p["composition_counts"][c] for c in classes}
    n = sum(totals.values())
    share: dict[str, float] = {}
    for c, k in totals.items():
        share[p["class_map"][c]] = share.get(p["class_map"][c], 0.0) + k / n * 100
    category_factor = {}
    for cat, (f5, f10) in p["factors"].items():
        s = share.get(cat, 0.0)
        category_factor[cat] = round(f5 + (f10 - f5) * min(max((s - 5) / 5, 0), 1), 4)
    class_factor = {c: category_factor[p["class_map"][c]] for c in classes}
    assert class_factor == A1_CLASS_FACTOR, f"PCU factors differ from Assignment 1: {class_factor}"
    return {
        "source": "IRC:106-1990 Table 1, factors at 5% and >=10% share, linear interpolation (A1)",
        "category_share_pct": {k: round(v, 2) for k, v in share.items()},
        "category_factor": category_factor,
        "class_category": p["class_map"],
        "class_factor": class_factor,
        "equals_assignment1": True,
    }


def categories(cfg: dict[str, Any]) -> list[str]:
    """IRC categories in config order."""
    return list(cfg["pcu"]["factors"])


def class_to_category(cfg: dict[str, Any]) -> np.ndarray:
    """(14,) IRC category index for each class index."""
    cats = categories(cfg)
    return np.array([cats.index(cfg["pcu"]["class_map"][c]) for c in cfg["dataset"]["classes"]])


def class_factor_vector(cfg: dict[str, Any]) -> np.ndarray:
    """(14,) PCU factor per class index."""
    f = pcu_factors(cfg)["class_factor"]
    return np.array([f[c] for c in cfg["dataset"]["classes"]])


def per_image_pcu(stems: np.ndarray, cls: np.ndarray, factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unique stems, PCU per image) from one class index per crop."""
    u, inv = np.unique(stems, return_inverse=True)
    return u, np.bincount(inv, weights=factor[cls], minlength=len(u))


def pcu_errors(stems: np.ndarray, true: np.ndarray, pred: np.ndarray, factor: np.ndarray) -> dict[str, float]:
    """Per-image PCU error from crop classes (boxes are ground truth, so counts are exact).

    pcu_mae: mean |pred PCU - true PCU| per image; pcu_rel_mae: that over mean true PCU per image (A1).
    """
    _, t = per_image_pcu(stems, true, factor)
    _, p = per_image_pcu(stems, pred, factor)
    err = p - t
    return {"pcu_mae": float(np.abs(err).mean()), "pcu_bias": float(err.mean()),
            "pcu_rel_mae": float(np.abs(err).mean() / t.mean()), "mean_true_pcu": float(t.mean()),
            "images": int(len(t))}
