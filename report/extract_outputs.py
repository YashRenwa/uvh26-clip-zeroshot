"""Pull every number the report quotes into report/numbers.json (flat keys, raw values).

The report text uses only these values; report/check_numbers.py (later) compares the report HTML with
this file. Sources are results/*.json and analysis/*.csv, written by the notebook run.

    python report/extract_outputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"

# Assignment 1 (fine-tuned YOLOv8, test, includes detection errors): quoted only as
# "what labelled training bought" (A1 report, ../uvh26-pcu-counting/Yash_Renwa_2025AG05357_EC1_Report.pdf).
A1_REFERENCE = {"a1.map50_95": 0.489, "a1.pcu_mae": 2.61, "a1.pcu_rel_mae": 0.158}


def load(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def flatten(prefix: str, d, out: dict) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(d, (int, float, str, bool)) or d is None:
        out[prefix] = d


def main() -> None:
    n: dict = dict(A1_REFERENCE)
    prov, data = load("provenance"), load("data_summary")
    n |= {"test_sha256": prov["test_sha256"], "model": prov["run"]["model"],
          "model_revision": prov["run"]["model_revision"], "dataset_revision": prov["run"]["dataset_revision"],
          "seed": prov["run"]["seed"]}
    flatten("split", prov["split"], n)
    flatten("leak", {k: v for k, v in prov["test_val_leakage"].items() if k not in ("same_camera_pairs",)}, n)
    n["leak.ci95.lo"], n["leak.ci95.hi"] = prov["test_val_leakage"]["leakage_95ci"]
    pairs = pd.read_csv(R / "leak_test_val" / "pairs.csv")
    for rank in (1, 12, 13):
        n[f"leak.pair{rank}.g"] = float(pairs.loc[pairs["rank"] == rank, "g"].iloc[0])
    for s in ("val", "test"):
        for k in ("images", "raw_boxes", "exact_duplicates_dropped", "crops", "groups", "ir_crops", "truncated_crops"):
            n[f"data.{s}.{k}"] = data[s][k]
        flatten(f"data.{s}.class_counts", data["class_counts"][s], n)
    n["data.low_support_classes"] = ", ".join(data["test"]["low_support_classes"])

    keys = ["accuracy", "balanced_accuracy", "macro_f1_headline", "macro_f1_all_present", "irc_accuracy",
            "irc_accuracy_from_class_argmax"]
    conf = load("val_confidence")
    for arm in ("baseline", "ensemble"):
        v = load(f"val_{arm}")
        for k in keys:
            n[f"val.{arm}.{k}"] = v[k]
        flatten(f"val.{arm}.pcu", v["pcu"], n)
        flatten(f"val.{arm}.pcu_argmax", v["pcu_from_class_argmax"], n)
        c = conf["arms"][arm]
        for lvl in ("class", "irc"):
            n[f"val.{arm}.{lvl}.ece"], n[f"val.{arm}.{lvl}.ece_equal_mass"] = c[lvl]["ece"], c[lvl]["ece_equal_mass"]
        r = c["rule"]
        for k in ("threshold", "coverage", "accepted_irc_accuracy", "accepted_class_accuracy", "human_workload"):
            n[f"val.{arm}.rule.{k}"] = r[k]
        n[f"val.{arm}.rule.pcu_mae_after_review"] = r["pcu_after_review"]["pcu_mae"]
        for tgt, x in c["rule_context_only"].items():
            n[f"val.{arm}.rule_context.{tgt}.coverage"] = x["coverage"]
        n[f"val.{arm}.ms_per_crop"] = v["timing"]["ms_per_crop"]
        n[f"val.{arm}.peak_rss_mb"] = v["timing"]["peak_rss_mb"]

    t = load("test_metrics")
    n |= {"test.crops": t["n_crops"], "test.images": t["n_images"], "test.groups": t["n_groups"]}
    for arm in ("baseline", "ensemble"):
        a = t["arms"][arm]
        for k in keys:
            n[f"test.{arm}.{k}"] = a[k]
        flatten(f"test.{arm}.pcu", a["pcu"], n)
        flatten(f"test.{arm}.pcu_argmax", a["pcu_from_class_argmax"], n)
        for lvl in ("class_confidence", "irc_confidence"):
            short = lvl.split("_")[0]
            n[f"test.{arm}.{short}.ece"] = a[lvl]["ece"]
            n[f"test.{arm}.{short}.ece_equal_mass"] = a[lvl]["ece_equal_mass"]
        r = a["rule"]
        for k in ("threshold", "coverage", "accepted_irc_accuracy", "accepted_class_accuracy", "human_workload"):
            n[f"test.{arm}.rule.{k}"] = r[k]
        n[f"test.{arm}.rule.pcu_mae_before_review"] = r["pcu_no_review"]["pcu_mae"]
        n[f"test.{arm}.rule.pcu_mae_after_review"] = r["pcu_after_review"]["pcu_mae"]
        for c, pc in a["per_class"].items():
            for k in ("precision", "recall", "f1", "support"):
                n[f"test.{arm}.class.{c}.{k}"] = pc[k]
        for k, ci in t["bootstrap"]["per_arm_ci95"][arm].items():
            n[f"test.{arm}.ci95.{k}.lo"], n[f"test.{arm}.ci95.{k}.hi"] = ci
    for k, d in t["bootstrap"]["ensemble_minus_baseline"].items():
        n[f"test.diff.{k}.lo"], n[f"test.diff.{k}.hi"] = d["ci95"]
        n[f"test.diff.{k}.share_above_0"] = d["share_resamples_above_0"]
    n["test.bootstrap.resamples"] = t["bootstrap"]["resamples"]

    fp = load("failure_patterns")
    for k in ("class_errors", "irc_errors", "accepted", "accepted_irc_wrong", "accepted_class_wrong_irc_right"):
        n[f"fail.{k}"] = fp[k]
    tags = pd.read_csv(ROOT / "analysis" / "failure_tags.csv")
    for s, g in tags.groupby("set"):
        for cause, cnt in g["cause"].value_counts().items():
            n[f"fail.tags.{s}.{cause}"] = int(cnt)
    rnd = tags[tags["set"] == "random"]["cause"]
    n["fail.tags.random.fills_crop"] = int(rnd.isin(["occlusion", "context", "truncated", "small"]).sum())
    for table, recs in fp["patterns"].items():
        for rec in recs:
            key = str(list(rec.values())[0])
            for k in ("crops", "irc_errors", "irc_error_rate", "class_error_rate"):
                n[f"fail.pattern.{table}.{key}.{k}"] = rec[k]

    # per-class PCU error mass of the headline arm on test (summed-softmax IRC prediction)
    rule = load("frozen_rule")
    arm = rule["headline_arm"]
    t_arm = rule["rules"][arm]["threshold"]
    from src.pcu import pcu_factors
    full_cfg = __import__("src.common", fromlist=["load_config"]).load_config()
    cf = pcu_factors(full_cfg)["category_factor"]
    cmap = full_cfg["pcu"]["class_map"]
    tp = pd.read_csv(R / "test_predictions.csv")
    err = tp[f"{arm}_irc_pred"].map(cf) - tp["true"].map(cmap).map(cf)
    total = float(err.abs().sum())
    n["pcu_mass.total_abs"] = total
    for c, g in err.groupby(tp["true"]):
        n[f"pcu_mass.{c}.abs"] = float(g.abs().sum())
        n[f"pcu_mass.{c}.net"] = float(g.sum())
        n[f"pcu_mass.{c}.share"] = float(g.abs().sum() / total)
        n[f"pcu_mass.{c}.crop_share"] = float(len(g) / len(tp))
    acc = tp[f"{arm}_irc_score"] >= t_arm
    wrong = tp[f"{arm}_irc_pred"] != tp["true"].map(cmap)
    for c in ("Truck", "LCV"):
        n[f"accepted.{c}.crops"] = int((acc & (tp["true"] == c)).sum())
        n[f"accepted.{c}.irc_wrong"] = int((acc & wrong & (tp["true"] == c)).sum())
    for k in ("truck_bus", "lcv", "auto", "car", "two_wheeler"):
        n[f"pcu_factor.{k}"] = cf[k]

    lc = load("label_check")
    flatten("dup", {s: {k: v for k, v in d.items() if k != "group_size_distribution"} for s, d in lc["splits"].items()}, n)
    for arm, r in lc["test"].items():
        a = r["a_excluding_duplicate_groups"]
        n |= {f"dup.{arm}.excl.accuracy": a["accuracy"], f"dup.{arm}.excl.irc_accuracy": a["irc_accuracy"],
              f"dup.{arm}.excl.pcu_mae": a["pcu"]["pcu_mae"], f"dup.{arm}.excl.accepted_irc_accuracy": a["rule"]["accepted_irc_accuracy"],
              f"dup.{arm}.lenient.accuracy": r["b_lenient_all_crops"]["accuracy"],
              f"dup.{arm}.lenient.irc_accuracy": r["b_lenient_all_crops"]["irc_accuracy"]}

    log = pd.read_csv(R / "prompt_log.csv")
    for rec in log.to_dict("records"):
        for k in ("irc_accuracy", "pcu_mae", "accuracy"):
            n[f"prompt_log.{rec['version']}.{k}"] = rec[k]

    out = ROOT / "report" / "numbers.json"
    out.write_text(json.dumps(n, indent=1, sort_keys=True))
    print(f"wrote {out} ({len(n)} numbers)")


if __name__ == "__main__":
    main()
