"""Sealed test evaluation (Phase 5): both frozen arms on the same test crops, opened once.

    EVALUATE_TEST=1 python -m src.evaluate_test        # the real run
    python -m src.evaluate_test --smoke                 # same code on 20 val images (never test)

Guards (all must pass before a test crop is scored): EVALUATE_TEST=1, splits/test.txt sha256 matches,
both prompt YAMLs match their sha in results/prompts_frozen.json, results/frozen_rule.json exists and
names the same prompt shas. Every opening is logged to results/test_opened.json.

Outputs: results/test_metrics.json, results/test_predictions.csv (crop_id, true, and per arm: pred, score,
IRC pred, IRC score; no embeddings), results/test_confusion_*.csv, results/figures/test_{reliability,coverage}.png.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from src.common import ROOT, add_common_args, load_config, set_seed, verify_test_split, write_result
from src.confidence import apply_rule, coverage_curve, ece, reliability
from src.metrics import classification_report, confusion, irc_prediction
from src.pcu import categories, class_factor_vector, class_to_category, pcu_errors, pcu_factors, per_image_pcu

ARMS = ("baseline", "ensemble")
FROZEN_PROMPTS = ROOT / "results" / "prompts_frozen.json"
FROZEN_RULE = ROOT / "results" / "frozen_rule.json"
OPENED = ROOT / "results" / "test_opened.json"


def guard() -> dict[str, Any]:
    """Raise unless every precondition for opening test holds."""
    if os.environ.get("EVALUATE_TEST") != "1":
        raise PermissionError("test is sealed: set EVALUATE_TEST=1 to open it")
    cfg = load_config()
    test_sha = verify_test_split(cfg)
    if not (FROZEN_PROMPTS.exists() and FROZEN_RULE.exists()):
        raise PermissionError("prompts_frozen.json and frozen_rule.json must exist before test is opened")
    prompts = json.loads(FROZEN_PROMPTS.read_text())["prompts"]
    rule = json.loads(FROZEN_RULE.read_text())
    for arm in ARMS:
        sha = hashlib.sha256((ROOT / "prompts" / f"{arm}.yaml").read_bytes()).hexdigest()
        if sha != prompts[arm]["sha256"] or sha != rule["rules"][arm]["prompt_sha256"]:
            raise PermissionError(f"{arm}.yaml does not match its frozen sha256")
    return {"test_sha256": test_sha, "prompt_sha256": {a: prompts[a]["sha256"] for a in ARMS},
            "thresholds": {a: rule["rules"][a]["threshold"] for a in ARMS}}


def log_opening(info: dict[str, Any], note: str) -> None:
    log = json.loads(OPENED.read_text()) if OPENED.exists() else {"openings": []}
    log["openings"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "note": note, **info})
    OPENED.write_text(json.dumps(log, indent=2))


# ---------- weighted statistics for the group bootstrap ----------

class ArmStats:
    """Per-crop outcomes of one arm, so any resample is a weighted sum."""

    def __init__(self, cfg, true, sm, stems, threshold, low_support):
        classes = cfg["dataset"]["classes"]
        self.k = len(classes)
        self.true, self.pred = true, sm.argmax(1)
        c2k = class_to_category(cfg)
        self.irc_true = c2k[true]
        self.irc_pred, self.irc_score = irc_prediction(cfg, sm)
        self.headline = ~np.isin(classes, low_support)
        cat_factor = np.array([pcu_factors(cfg)["category_factor"][c] for c in categories(cfg)])
        u, t = per_image_pcu(stems, self.irc_true, cat_factor)
        _, p = per_image_pcu(stems, self.irc_pred, cat_factor)
        self.img_abs_err = np.abs(p - t)
        self.img_index = np.searchsorted(u, stems)            # crop -> image
        self.accept = self.irc_score >= threshold

    def stats(self, w: np.ndarray, w_img: np.ndarray) -> dict[str, float]:
        cm = np.bincount(self.true * self.k + self.pred, weights=w, minlength=self.k ** 2).reshape(self.k, self.k)
        tp, sup, prd = np.diag(cm), cm.sum(1), cm.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(prd > 0, tp / prd, 0.0)
            rec = np.where(sup > 0, tp / sup, 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        keep = self.headline & (sup > 0)
        irc_ok = self.irc_pred == self.irc_true
        wa = w * self.accept
        return {"accuracy": float((w * (self.true == self.pred)).sum() / w.sum()),
                "macro_f1_headline": float(f1[keep].mean()),
                "irc_accuracy": float((w * irc_ok).sum() / w.sum()),
                "pcu_mae": float((w_img * self.img_abs_err).sum() / w_img.sum()),
                "accepted_irc_accuracy": float((wa * irc_ok).sum() / wa.sum()) if wa.sum() > 0 else np.nan}


def group_bootstrap(cfg, arms: dict[str, ArmStats], groups: np.ndarray, img_groups: np.ndarray) -> dict[str, Any]:
    """Paired percentile bootstrap over pseudo-camera groups: per-arm CIs and ensemble - baseline CIs."""
    n = cfg["bootstrap"]["n_resamples"]
    rng = np.random.default_rng(cfg["seed"])
    ug, g_idx = np.unique(groups, return_inverse=True)
    img_g_idx = np.searchsorted(ug, img_groups)
    keys = ["accuracy", "macro_f1_headline", "irc_accuracy", "pcu_mae", "accepted_irc_accuracy"]
    draws = {a: {k: np.empty(n) for k in keys} for a in arms}
    for b in range(n):
        mult = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
        w, w_img = mult[g_idx], mult[img_g_idx]
        for a, s in arms.items():
            for k, v in s.stats(w, w_img).items():
                draws[a][k][b] = v

    def ci(x):
        return [round(float(np.nanpercentile(x, 2.5)), 4), round(float(np.nanpercentile(x, 97.5)), 4)]

    out = {"resamples": n, "unit": "pseudo-camera group", "groups": int(len(ug)), "seed": cfg["seed"],
           "per_arm_ci95": {a: {k: ci(draws[a][k]) for k in keys} for a in arms}}
    if set(arms) == set(ARMS):
        diff = {k: draws["ensemble"][k] - draws["baseline"][k] for k in keys[:4]}
        out["ensemble_minus_baseline"] = {k: {"ci95": ci(d), "share_resamples_above_0": round(float((d > 0).mean()), 4)}
                                          for k, d in diff.items()}
    return out


# ---------- one arm, point estimates ----------

def arm_report(cfg, arm, true, sm, stems, threshold, low_support) -> dict[str, Any]:
    nb = cfg["confidence"]["n_bins"]
    pred, score = sm.argmax(1), sm.max(1)
    rep = classification_report(cfg, true, pred, sm, low_support)
    c2k = class_to_category(cfg)
    irc_pred, irc_score = irc_prediction(cfg, sm)
    irc_true = c2k[true]
    correct, irc_ok = pred == true, irc_pred == irc_true
    cat_factor = np.array([pcu_factors(cfg)["category_factor"][c] for c in categories(cfg)])
    cm = np.array(rep["confusion"], float)
    cmi = np.array(rep["confusion_irc"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rep["confusion_row_normalised"] = np.round(cm / cm.sum(1, keepdims=True), 4).tolist()
        rep["confusion_irc_row_normalised"] = np.round(cmi / cmi.sum(1, keepdims=True), 4).tolist()
    return {
        **rep,
        "pcu": pcu_errors(stems, irc_true, irc_pred, cat_factor),
        "pcu_from_class_argmax": pcu_errors(stems, true, pred, class_factor_vector(cfg)),
        "class_confidence": {"mean_score": float(score.mean()), "ece": ece(score, correct, nb),
                             "ece_equal_mass": ece(score, correct, nb, True),
                             "reliability": reliability(score, correct, nb), "coverage_curve": coverage_curve(score, correct)},
        "irc_confidence": {"mean_score": float(irc_score.mean()), "ece": ece(irc_score, irc_ok, nb),
                           "ece_equal_mass": ece(irc_score, irc_ok, nb, True),
                           "reliability": reliability(irc_score, irc_ok, nb), "coverage_curve": coverage_curve(irc_score, irc_ok)},
        "rule": {"threshold_frozen": threshold, **apply_rule(cfg, threshold, stems, true, sm)},
    }


# ---------- figures: test with val as a thin line ----------

def plot_shift(test: dict, val: dict, out_dir, tag: str) -> list[str]:
    import matplotlib.pyplot as plt

    from src.plotting import ARM_COLOR, INK_2, REFERENCE, style

    style()
    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for col, (lvl_t, lvl_v, title) in enumerate((("class_confidence", "class", "14 classes: max softmax score"),
                                                  ("irc_confidence", "irc", "IRC category: summed softmax score"))):
        ax, axc = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="score = accuracy")
        for k, arm in enumerate(ARMS):
            if arm not in test:
                continue
            for src, lw, alpha, lab in ((val["arms"][arm][lvl_v]["reliability"], 1, 0.55, f"{arm} val"),
                                        (test[arm][lvl_t]["reliability"], 2, 1.0, None)):
                ms, acc = np.array(src["mean_score"], float), np.array(src["accuracy"], float)
                keep = ~np.isnan(ms)
                ax.plot(ms[keep], acc[keep], marker="o" if lw == 2 else None, ms=4.5, lw=lw, alpha=alpha,
                        color=ARM_COLOR[arm],
                        label=lab or f"{arm} {tag} (ECE {test[arm][lvl_t]['ece']:.3f})")
            rel = test[arm][lvl_t]["reliability"]
            edges = np.array(rel["edges"])
            width = 1 / len(rel["count"]) / 2.5
            axc.bar((edges[:-1] + edges[1:]) / 2 + (k - 0.5) * width, rel["count"], width=width * 0.9, color=ARM_COLOR[arm])
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_title(title), ax.set_ylabel("accuracy in bin")
        ax.legend(loc="upper left", fontsize=8)
        axc.set_xlabel("confidence score (not a probability)"), axc.set_ylabel(f"{tag} crops", color=INK_2)
    fig.suptitle(f"Reliability on {tag} (thick, dots) vs val (thin)", fontsize=11)
    fig.tight_layout()
    p = out_dir / f"{tag}_reliability.png"
    fig.savefig(p, dpi=130), plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (lvl_t, lvl_v, title) in zip(axes, (("class_confidence", "class", "14-class accuracy of accepted crops"),
                                                ("irc_confidence", "irc", "IRC accuracy of accepted crops"))):
        for arm in ARMS:
            if arm not in test:
                continue
            vc, tc = val["arms"][arm][lvl_v]["coverage_curve"], test[arm][lvl_t]["coverage_curve"]
            ax.plot(vc["coverage"], vc["accuracy"], color=ARM_COLOR[arm], lw=1, alpha=0.55, label=f"{arm} val")
            ax.plot(tc["coverage"], tc["accuracy"], color=ARM_COLOR[arm], lw=2, label=f"{arm} {tag}")
            if lvl_v == "irc":
                r = test[arm]["rule"]
                ax.plot([r["coverage"]], [r["accepted_irc_accuracy"]], "o", ms=8, color=ARM_COLOR[arm], mec="#fcfcfb", mew=2)
                ax.annotate(f"frozen t = {r['threshold']:.3f}\n{100 * r['coverage']:.0f}% auto, "
                            f"{100 * r['accepted_irc_accuracy']:.1f}%", (r["coverage"], r["accepted_irc_accuracy"]),
                            textcoords="offset points", xytext=(8, -30), fontsize=8)
        if lvl_v == "irc":
            ax.axhline(0.95, color=REFERENCE, lw=1, ls="--", label="target 95%")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.01), ax.set_title(title)
        ax.set_xlabel("coverage: share of crops accepted, most confident first")
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    p = out_dir / f"{tag}_coverage.png"
    fig.savefig(p, dpi=130), plt.close(fig)
    paths.append(str(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--note", default="Phase 5 sealed test run")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(cfg["seed"])
    from src.run_val import low_support_classes, val_manifest
    from src.zeroshot import (embedding_fingerprint, image_embeddings, load_model, logit_scale, predict,
                              text_classifier)

    if args.smoke:     # exercise the code on val images; test is never read
        split, manifest = "val_smoke", val_manifest(cfg)
        rule = json.loads(FROZEN_RULE.read_text())
        info = {"thresholds": {a: rule["rules"][a]["threshold"] for a in ARMS}}
        cache = cfg["paths"]["cache_dir"] / "img_emb_val_smoke.npz"
    else:
        info = guard()
        split = "test"
        manifest = pd.read_csv(ROOT / "results" / "crops_test.csv", dtype={"stem": str, "group": str})
        log_opening(info, args.note)
        cache = None   # test embeddings stay in memory
    res_dir = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]
    low = low_support_classes()

    model, processor = load_model(cfg)
    emb, timing = image_embeddings(cfg, model, processor, manifest, cache=cache)
    emb_sha = embedding_fingerprint(emb)
    true, stems = manifest["cls"].to_numpy(), manifest["stem"].to_numpy()
    out = {"split": split, "n_crops": len(manifest), "n_images": int(manifest["stem"].nunique()),
           "n_groups": int(manifest["group"].nunique()), "timing": timing, "image_embeddings_sha256": emb_sha,
           "image_embeddings_shared_by_arms": True, "low_support_classes": low, "arms": {}, **info}
    preds = {"crop_id": manifest["crop_id"], "true": manifest["class"]}
    stats = {}
    for arm in ARMS:
        cls_emb, pinfo = text_classifier(cfg, model, processor, ROOT / "prompts" / f"{arm}.yaml")
        assert embedding_fingerprint(emb) == emb_sha    # same image side for both arms
        _, _, sm = predict(emb, cls_emb, logit_scale(model))
        t = info["thresholds"][arm]
        out["arms"][arm] = {"prompt_sha256": pinfo["sha256"], **arm_report(cfg, arm, true, sm, stems, t, low)}
        irc_pred, irc_score = irc_prediction(cfg, sm)
        preds |= {f"{arm}_pred": np.array(classes)[sm.argmax(1)], f"{arm}_score": sm.max(1).round(5),
                  f"{arm}_irc_pred": np.array(categories(cfg))[irc_pred], f"{arm}_irc_score": irc_score.round(5)}
        stats[arm] = ArmStats(cfg, true, sm, stems, t, low)
        pd.DataFrame(out["arms"][arm]["confusion"], index=classes, columns=classes).to_csv(
            res_dir / f"{split}_confusion_{arm}.csv")
        pd.DataFrame(out["arms"][arm]["confusion_irc"], index=categories(cfg), columns=categories(cfg)).to_csv(
            res_dir / f"{split}_confusion_irc_{arm}.csv")

    img = manifest.drop_duplicates("stem").set_index("stem")["group"]
    u = np.unique(stems)
    out["bootstrap"] = group_bootstrap(cfg, stats, manifest["group"].to_numpy(), img.loc[u].to_numpy())
    val = json.loads((ROOT / "results" / "val_confidence.json").read_text())
    out["figures"] = plot_shift(out["arms"], val, res_dir / "figures", "test" if split == "test" else split)
    pd.DataFrame(preds).to_csv(res_dir / f"{split}_predictions.csv", index=False)
    write_result(f"{split}_metrics", out, cfg)
    assert not any(p.suffix in (".npy", ".npz") for p in res_dir.rglob("*")), "no embedding arrays in results/"
    for arm in ARMS:
        a = out["arms"][arm]
        r = a["rule"]
        print(f"{arm}: acc {a['accuracy']:.3f} IRC {a['irc_accuracy']:.3f} (argmax {a['irc_accuracy_from_class_argmax']:.3f}) "
              f"PCU {a['pcu']['pcu_mae']:.2f} | rule cov {r['coverage']:.3f} acc {r['accepted_irc_accuracy']:.3f}")
    print(json.dumps(out["bootstrap"].get("ensemble_minus_baseline"), indent=1))


if __name__ == "__main__":
    main()
