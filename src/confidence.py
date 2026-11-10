"""Confidence vs correctness on val, and the review rule (Phase 4).

Score = max softmax over the 14 classes (a confidence score, not a probability).
IRC level: IRC prediction = category with the highest summed softmax; IRC score = that sum. PCU depends
only on the IRC category, so the review rule works at IRC level:
    t = lowest IRC score at which the accepted crops (IRC score >= t) reach the target IRC accuracy.
Crops below t go to a human; we model the human as correct (their class is set to the truth).

    python -m src.confidence --arms baseline [ensemble] [--smoke]

Writes results/val_confidence.json and results/figures/val_{reliability,coverage}.png.
The ensemble arm only runs once its prompt YAML is frozen (G1). The rule is frozen separately at G2
(freeze_rule -> results/frozen_rule.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.common import ROOT, add_common_args, load_config, set_seed, write_result
from src.metrics import irc_prediction
from src.pcu import categories, class_factor_vector, class_to_category, pcu_errors, pcu_factors

RULE_PATH = ROOT / "results" / "frozen_rule.json"


# ---------- calibration ----------

def reliability(score: np.ndarray, correct: np.ndarray, n_bins: int, equal_mass: bool = False) -> dict[str, list]:
    """Per-bin count, mean score and accuracy. Equal-width bins on [0, 1], or equal-mass (quantile) bins."""
    if equal_mass:
        edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    s_sum = np.bincount(idx, weights=score, minlength=n_bins)
    c_sum = np.bincount(idx, weights=correct.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_score, acc = s_sum / count, c_sum / count
    return {"edges": edges.round(4).tolist(), "count": count.tolist(),
            "mean_score": np.where(count > 0, mean_score, np.nan).round(4).tolist(),
            "accuracy": np.where(count > 0, acc, np.nan).round(4).tolist()}


def ece(score: np.ndarray, correct: np.ndarray, n_bins: int, equal_mass: bool = False) -> float:
    """Expected calibration error: sum over bins of (bin share) x |accuracy - mean score|."""
    r = reliability(score, correct, n_bins, equal_mass)
    c = np.array(r["count"])
    gap = np.abs(np.nan_to_num(np.array(r["accuracy"], float)) - np.nan_to_num(np.array(r["mean_score"], float)))
    return float((c * gap).sum() / c.sum())


def coverage_curve(score: np.ndarray, correct: np.ndarray, points: int = 101) -> dict[str, list]:
    """Accuracy of the top-c most confident crops, for coverage c on a grid from 1% to 100%."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    cov = np.linspace(0.01, 1.0, points)
    k = np.clip(np.ceil(cov * len(order)).astype(int), 1, len(order))
    return {"coverage": cov.round(4).tolist(), "accuracy": cum[k - 1].round(4).tolist(),
            "score_at": score[order][k - 1].round(4).tolist()}


# ---------- review rule ----------

def review_threshold(irc_score: np.ndarray, irc_correct: np.ndarray, target: float) -> float | None:
    """Lowest t such that crops with irc_score >= t have IRC accuracy >= target (None if unreachable)."""
    order = np.argsort(-irc_score, kind="stable")
    s = irc_score[order]
    acc = np.cumsum(irc_correct[order]) / np.arange(1, len(s) + 1)
    # Only cut between distinct scores: a threshold accepts every crop with that score.
    last_of_value = np.r_[s[1:] != s[:-1], True]
    ok = np.nonzero((acc >= target) & last_of_value)[0]
    return None if len(ok) == 0 else float(s[ok.max()])


def apply_rule(cfg: dict[str, Any], t: float | None, stems: np.ndarray, true_cls: np.ndarray,
               softmax: np.ndarray) -> dict[str, Any]:
    """Coverage, accepted accuracy, workload and PCU error with human review below t."""
    c2k = class_to_category(cfg)
    cat_factor = np.array([pcu_factors(cfg)["category_factor"][c] for c in categories(cfg)])
    irc_true = c2k[true_cls]
    irc_pred, irc_score = irc_prediction(cfg, softmax)
    cls_pred = softmax.argmax(1)
    accept = np.zeros(len(true_cls), bool) if t is None else irc_score >= t
    reviewed_irc = np.where(accept, irc_pred, irc_true)
    out = {
        "threshold": t,
        "coverage": float(accept.mean()),
        "human_workload": float(1 - accept.mean()),
        "accepted": int(accept.sum()),
        "accepted_irc_accuracy": float((irc_pred[accept] == irc_true[accept]).mean()) if accept.any() else None,
        "accepted_class_accuracy": float((cls_pred[accept] == true_cls[accept]).mean()) if accept.any() else None,
        "pcu_no_review": pcu_errors(stems, irc_true, irc_pred, cat_factor),
        "pcu_after_review": pcu_errors(stems, irc_true, reviewed_irc, cat_factor),
        "pcu_no_review_from_class_argmax": pcu_errors(stems, true_cls, cls_pred, class_factor_vector(cfg)),
    }
    return out


# ---------- per arm ----------

def arm_outputs(cfg: dict[str, Any], arm: str) -> dict[str, np.ndarray]:
    """Val softmax for one frozen arm from the shared image embeddings (dev cache)."""
    from src.run_val import val_manifest
    from src.zeroshot import image_embeddings, load_model, logit_scale, predict, text_classifier

    manifest = val_manifest(cfg)
    model, processor = load_model(cfg)
    cache = cfg["paths"]["cache_dir"] / f"img_emb_val{'_smoke' if cfg['smoke'] else ''}.npz"
    emb, _ = image_embeddings(cfg, model, processor, manifest, cache=cache)
    cls_emb, info = text_classifier(cfg, model, processor, ROOT / "prompts" / f"{arm}.yaml")   # frozen only
    _, _, sm = predict(emb, cls_emb, logit_scale(model))
    from src.zeroshot import embedding_fingerprint

    return {"stems": manifest["stem"].to_numpy(), "true": manifest["cls"].to_numpy(), "softmax": sm,
            "prompt_sha256": info["sha256"], "image_embeddings_sha256": embedding_fingerprint(emb)}


def analyse(cfg: dict[str, Any], o: dict[str, np.ndarray]) -> dict[str, Any]:
    nb, target = cfg["confidence"]["n_bins"], cfg["confidence"]["target_irc_accuracy"]
    sm, true = o["softmax"], o["true"]
    pred, score = sm.argmax(1), sm.max(1)
    irc_pred, irc_score = irc_prediction(cfg, sm)
    irc_true = class_to_category(cfg)[true]
    correct, irc_correct = pred == true, irc_pred == irc_true
    t = review_threshold(irc_score, irc_correct, target)
    argmax_irc_correct = class_to_category(cfg)[pred] == irc_true
    context = {}
    for tgt in cfg["confidence"]["context_targets"]:
        tc = review_threshold(irc_score, irc_correct, tgt)
        r = apply_rule(cfg, tc, o["stems"], true, sm)
        context[f"{tgt:.2f}"] = {k: r[k] for k in ("threshold", "coverage", "human_workload", "accepted_irc_accuracy")}
    return {
        "prompt_sha256": o["prompt_sha256"], "image_embeddings_sha256": o["image_embeddings_sha256"],
        "n": int(len(true)),
        "class": {"accuracy": float(correct.mean()), "mean_score": float(score.mean()),
                  "ece": ece(score, correct, nb), "ece_equal_mass": ece(score, correct, nb, True),
                  "reliability": reliability(score, correct, nb), "coverage_curve": coverage_curve(score, correct)},
        "irc": {"accuracy": float(irc_correct.mean()),
                "accuracy_from_class_argmax": float(argmax_irc_correct.mean()),
                "mean_score": float(irc_score.mean()),
                "ece": ece(irc_score, irc_correct, nb), "ece_equal_mass": ece(irc_score, irc_correct, nb, True),
                "reliability": reliability(irc_score, irc_correct, nb),
                "coverage_curve": coverage_curve(irc_score, irc_correct)},
        "rule": {"target_irc_accuracy": target, **apply_rule(cfg, t, o["stems"], true, sm)},
        "rule_context_only": context,
    }


# ---------- figures ----------

def plot_reliability(res: dict[str, Any], out: Path) -> Path:
    import matplotlib.pyplot as plt

    from src.plotting import ARM_COLOR, INK_2, REFERENCE, style

    style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for col, level in enumerate(("class", "irc")):
        ax, axc = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], color=REFERENCE, lw=1, ls="--", label="score = accuracy")
        arms = list(res["arms"])
        width = 1 / res["n_bins"] / (len(arms) + 0.5)
        for k, (arm, r) in enumerate(res["arms"].items()):
            rel = r[level]["reliability"]
            ms, acc = np.array(rel["mean_score"], float), np.array(rel["accuracy"], float)
            keep = ~np.isnan(ms)
            ax.plot(ms[keep], acc[keep], marker="o", ms=4.5, color=ARM_COLOR[arm],
                    label=f"{arm}  (ECE {r[level]['ece']:.3f})")
            edges = np.array(rel["edges"])
            centres = (edges[:-1] + edges[1:]) / 2 + (k - (len(arms) - 1) / 2) * width
            axc.bar(centres, rel["count"], width=width * 0.9, color=ARM_COLOR[arm])
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy in bin")
        ax.set_title("14 classes: max softmax score" if level == "class" else "IRC category: summed softmax score")
        ax.legend(loc="upper left")
        axc.set_xlabel("confidence score (not a probability)")
        axc.set_ylabel("crops", color=INK_2)
    fig.suptitle(f"Val reliability, {res['n_bins']} equal-width bins", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_coverage(res: dict[str, Any], out: Path) -> Path:
    import matplotlib.pyplot as plt

    from src.plotting import ARM_COLOR, REFERENCE, style

    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, level in zip(axes, ("class", "irc")):
        for arm, r in res["arms"].items():
            cc = r[level]["coverage_curve"]
            ax.plot(cc["coverage"], cc["accuracy"], color=ARM_COLOR[arm], label=arm)
            if level == "irc" and r["rule"]["threshold"] is not None:
                ax.plot([r["rule"]["coverage"]], [r["rule"]["accepted_irc_accuracy"]], "o", ms=8,
                        color=ARM_COLOR[arm], mec="#fcfcfb", mew=2)
                ax.annotate(f"t = {r['rule']['threshold']:.3f}\n{100 * r['rule']['coverage']:.0f}% auto",
                            (r["rule"]["coverage"], r["rule"]["accepted_irc_accuracy"]),
                            textcoords="offset points", xytext=(8, -26), fontsize=8.5)
        if level == "irc":
            ax.axhline(res["target_irc_accuracy"], color=REFERENCE, lw=1, ls="--",
                       label=f"target {res['target_irc_accuracy']:.0%}")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
        ax.set_xlabel("coverage: share of crops accepted, most confident first")
        ax.set_title("14-class accuracy of accepted crops" if level == "class" else "IRC accuracy of accepted crops")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------- freezing (G2) ----------

def freeze_rule(cfg: dict[str, Any], arm: str, by: str) -> dict[str, Any]:
    """Write results/frozen_rule.json from results/val_confidence.json (refuses to overwrite)."""
    if RULE_PATH.exists():
        raise PermissionError(f"{RULE_PATH.name} already exists; the rule is frozen")
    res = json.loads((ROOT / "results" / "val_confidence.json").read_text())
    r = res["arms"][arm]
    rule = {"arm": arm, "target_irc_accuracy": r["rule"]["target_irc_accuracy"], "threshold": r["rule"]["threshold"],
            "score": "IRC score = summed softmax of the IRC category's classes", "prompt_sha256": r["prompt_sha256"],
            "val": {k: r["rule"][k] for k in ("coverage", "accepted_irc_accuracy", "human_workload")},
            "decided_by": by, "frozen_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    RULE_PATH.write_text(json.dumps(rule, indent=2))
    return rule


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--arms", nargs="+", default=["baseline", "ensemble"])
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(cfg["seed"])
    res = {"split": "val", "n_bins": cfg["confidence"]["n_bins"],
           "target_irc_accuracy": cfg["confidence"]["target_irc_accuracy"], "arms": {}}
    for arm in args.arms:
        res["arms"][arm] = analyse(cfg, arm_outputs(cfg, arm))
    shas = {r["image_embeddings_sha256"] for r in res["arms"].values()}
    assert len(shas) == 1, "arms must share one set of image embeddings"
    res["image_embeddings_shared"] = True
    figs = cfg["paths"]["results_dir"] / "figures"
    res["figures"] = [str(plot_reliability(res, figs / "val_reliability.png")),
                      str(plot_coverage(res, figs / "val_coverage.png"))]
    write_result("val_confidence", res, cfg)
    for arm, r in res["arms"].items():
        rule = r["rule"]
        print(f"{arm}: class acc {r['class']['accuracy']:.3f} ECE {r['class']['ece']:.3f} "
              f"(equal-mass {r['class']['ece_equal_mass']:.3f}) | IRC acc {r['irc']['accuracy']:.3f} "
              f"ECE {r['irc']['ece']:.3f} (equal-mass {r['irc']['ece_equal_mass']:.3f})")
        if rule["threshold"] is None:
            print(f"  rule: target {rule['target_irc_accuracy']:.0%} not reachable")
        else:
            print(f"  rule: t {rule['threshold']:.4f} -> coverage {rule['coverage']:.3f}, accepted IRC acc "
                  f"{rule['accepted_irc_accuracy']:.3f}, workload {rule['human_workload']:.3f}, PCU MAE "
                  f"{rule['pcu_no_review']['pcu_mae']:.2f} -> {rule['pcu_after_review']['pcu_mae']:.2f} after review")


if __name__ == "__main__":
    main()
