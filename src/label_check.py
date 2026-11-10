"""SECONDARY label-quality check: STAPLE double-labelled vehicles (headline numbers unchanged).

Defined on 2026-11-10 AFTER test was opened, as a label-quality check, not a model choice (DECISIONS.md).
Duplicate group: same-image ST boxes linked by IoU >= iou_min with different classes (connected
components). Reported on the frozen test predictions, both arms:
  (a) excluding every crop in a duplicate group: metrics, PCU recomputed on the remaining crops for both
      true and predicted, and the frozen review rule's accepted IRC accuracy;
  (b) lenient accuracy: a crop is correct if its prediction matches ANY label in its group
      (14-class: any class; IRC: any category).
No model is re-run; test_predictions.csv is the input.

    python -m src.label_check
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from src.common import ROOT, add_common_args, load_config, write_result
from src.metrics import per_class
from src.pcu import categories, pcu_errors, pcu_factors

IOU_MIN = 0.99


def duplicate_groups(m: pd.DataFrame, iou_min: float = IOU_MIN) -> pd.Series:
    """Group id per crop (-1 if the crop is in no duplicate group)."""
    gid = pd.Series(-1, index=m.index)
    nxt = 0
    for _, g in m.groupby("stem", sort=False):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        iw = np.clip(np.minimum(b[:, None, 2], b[:, 2]) - np.maximum(b[:, None, 0], b[:, 0]), 0, None)
        ih = np.clip(np.minimum(b[:, None, 3], b[:, 3]) - np.maximum(b[:, None, 1], b[:, 1]), 0, None)
        inter = iw * ih
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area[:, None] + area - inter)
        cls = g["class"].to_numpy()
        link = (iou >= iou_min) & (cls[:, None] != cls)
        np.fill_diagonal(link, False)
        parent = list(range(len(g)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in zip(*np.nonzero(link)):
            parent[find(i)] = find(j)
        roots = np.array([find(i) for i in range(len(g))])
        linked = link.any(1)
        for r in np.unique(roots[linked]):
            gid.loc[g.index[(roots == r) & linked]] = nxt
            nxt += 1
    return gid


def arm_metrics(cfg, d: pd.DataFrame, arm: str, t: float, low: list[str]) -> dict[str, Any]:
    classes = cfg["dataset"]["classes"]
    cats = categories(cfg)
    cmap = cfg["pcu"]["class_map"]
    true = d["class"].map(classes.index).to_numpy()
    pred = d[f"{arm}_pred"].map(classes.index).to_numpy()
    irc_true = d["class"].map(lambda c: cats.index(cmap[c])).to_numpy()
    irc_pred = d[f"{arm}_irc_pred"].map(cats.index).to_numpy()
    k = len(classes)
    pc = per_class(np.bincount(true * k + pred, minlength=k * k).reshape(k, k))
    keep = (pc["support"] > 0) & ~np.isin(classes, low)
    cat_factor = np.array([pcu_factors(cfg)["category_factor"][c] for c in cats])
    acc = d[f"{arm}_irc_score"].to_numpy() >= t
    return {"crops": int(len(d)), "images": int(d["stem"].nunique()),
            "accuracy": float((true == pred).mean()), "macro_f1_headline": float(pc["f1"][keep].mean()),
            "irc_accuracy": float((irc_true == irc_pred).mean()),
            "pcu": pcu_errors(d["stem"].to_numpy(), irc_true, irc_pred, cat_factor),
            "rule": {"threshold": t, "coverage": float(acc.mean()),
                     "accepted_irc_accuracy": float((irc_true[acc] == irc_pred[acc]).mean())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rule = json.loads((ROOT / "results" / "frozen_rule.json").read_text())
    low = json.loads((ROOT / "results" / "data_summary.json").read_text())["test"]["low_support_classes"]
    cmap = cfg["pcu"]["class_map"]
    out: dict[str, Any] = {"status": "SECONDARY label-quality check, defined after test was opened; headline unchanged",
                           "definition": f"same-image ST boxes linked by IoU >= {IOU_MIN} with different classes",
                           "splits": {}}
    for split in ("val", "test"):
        m = pd.read_csv(ROOT / "results" / f"crops_{split}.csv", dtype={"stem": str, "group": str})
        m["dup_group"] = duplicate_groups(m)
        sizes = m.loc[m["dup_group"] >= 0].groupby("dup_group").size()
        out["splits"][split] = {"crops": len(m), "crops_in_groups": int((m["dup_group"] >= 0).sum()),
                                "share": round(float((m["dup_group"] >= 0).mean()), 4), "groups": int(len(sizes)),
                                "images_with_groups": int(m.loc[m["dup_group"] >= 0, "stem"].nunique()),
                                "group_size_distribution": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}}
        if split == "test":
            test_m = m
    pred = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    d = test_m.merge(pred.drop(columns=["true"]), on="crop_id", validate="one_to_one")
    in_dup = d["dup_group"] >= 0
    labels = d[in_dup].groupby("dup_group")["class"].agg(set)
    res = {}
    for arm in ("baseline", "ensemble"):
        t = rule["rules"][arm]["threshold"]
        full, excl = arm_metrics(cfg, d, arm, t, low), arm_metrics(cfg, d[~in_dup], arm, t, low)
        any_cls = d["dup_group"].map(labels).where(in_dup, d["class"].map(lambda c: {c}))
        len_ok = [p in s for p, s in zip(d[f"{arm}_pred"], any_cls)]
        len_irc = [p in {cmap[c] for c in s} for p, s in zip(d[f"{arm}_irc_pred"], any_cls)]
        res[arm] = {"all_crops_headline": {k: full[k] for k in ("accuracy", "macro_f1_headline", "irc_accuracy")}
                    | {"pcu_mae": full["pcu"]["pcu_mae"], "accepted_irc_accuracy": full["rule"]["accepted_irc_accuracy"]},
                    "a_excluding_duplicate_groups": excl,
                    "b_lenient_all_crops": {"accuracy": float(np.mean(len_ok)), "irc_accuracy": float(np.mean(len_irc))}}
    out["test"] = res
    out["note_a1"] = "A1's labels (same ST JSONs, same dedup rule) carry the same duplicates."
    write_result("label_check", out, cfg)
    print(json.dumps(out["splits"], indent=1))
    for arm, r in res.items():
        f, a, b = r["all_crops_headline"], r["a_excluding_duplicate_groups"], r["b_lenient_all_crops"]
        print(f"{arm}: acc {f['accuracy']:.3f} -> excl {a['accuracy']:.3f}, lenient {b['accuracy']:.3f} | IRC {f['irc_accuracy']:.3f} "
              f"-> excl {a['irc_accuracy']:.3f}, lenient {b['irc_accuracy']:.3f} | mF1 {f['macro_f1_headline']:.3f} -> {a['macro_f1_headline']:.3f} "
              f"| PCU {f['pcu_mae']:.2f} -> {a['pcu']['pcu_mae']:.2f} (mean true {a['pcu']['mean_true_pcu']:.2f}, bias {a['pcu']['pcu_bias']:+.2f}) "
              f"| accepted IRC {f['accepted_irc_accuracy']:.4f} -> {a['rule']['accepted_irc_accuracy']:.4f} (cov {a['rule']['coverage']:.3f}) | "
              f"crops {a['crops']} images {a['images']}")


if __name__ == "__main__":
    main()
