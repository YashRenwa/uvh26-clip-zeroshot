"""Report figures that are not written elsewhere, plus the redaction check list.

    python -m src.report_figures

report/figures/test_confusion_irc_ensemble.png   IRC-level confusion, headline arm, row-normalised
report/figures/failures_first5.png               the first five of the seeded random error draw, redacted
results/redaction_checklist.csv                  every image with pixels that the report or notebook shows
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.common import ROOT, load_config, write_result
from src.pcu import categories

FIG_DIR = ROOT / "report" / "figures"


def confusion_irc(cfg: dict[str, Any], arm: str = "ensemble") -> str:
    import matplotlib.pyplot as plt

    from src.plotting import INK, style

    style()
    t = json.loads((ROOT / "results" / "test_metrics.json").read_text())["arms"][arm]
    cm = np.array(t["confusion_irc"], float)
    rn = cm / cm.sum(1, keepdims=True)
    cats = [c.replace("_", " ") for c in categories(cfg)]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.grid(False)
    ax.imshow(rn, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#2a78d6", "#104281"]),
              vmin=0, vmax=1)
    for i in range(len(cats)):
        for j in range(len(cats)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{rn[i, j]:.2f}\n{int(cm[i, j])}", ha="center", va="center", fontsize=7.5,
                        color="#fcfcfb" if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cats)), cats, rotation=35, ha="right")
    ax.set_yticks(range(len(cats)), [f"{c} (n={int(n)})" for c, n in zip(cats, cm.sum(1))])
    ax.set_xlabel("predicted IRC category (summed softmax)"), ax.set_ylabel("true IRC category")
    ax.set_title(f"Test, {arm} arm: row share and crop count", fontsize=10.5)
    fig.tight_layout()
    out = FIG_DIR / f"test_confusion_irc_{arm}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150), plt.close(fig)
    return str(out.relative_to(ROOT))


def first_five_failures(cfg: dict[str, Any]) -> str:
    from src.failures import load_test, render_sheet

    tags = pd.read_csv(ROOT / "analysis" / "failure_tags.csv", dtype={"stem": str})
    five = tags[(tags["set"] == "random") & (tags["rank"].astype(int) <= 5)]
    rule = json.loads((ROOT / "results" / "frozen_rule.json").read_text())
    df = load_test(cfg, rule["headline_arm"], "test").set_index("crop_id")
    rows = df.loc[five["crop_id"]].reset_index()
    rows["set"], rows["rank"] = "random", five["rank"].astype(int).to_numpy()
    rows["suggested_cause"] = five["cause"].to_numpy()
    out = FIG_DIR / "failures_first5.png"
    render_sheet(cfg, rows, "First five of the seeded random error draw (ensemble, test); not chosen by hand", out,
                 cause_label="cause (Yash's tag)")
    return str(out.relative_to(ROOT))


def redaction_checklist(cfg: dict[str, Any], in_report: list[str]) -> pd.DataFrame:
    """Every pixel-showing image the report or notebook displays, with the Haar faces still detectable."""
    import cv2

    from src.privacy import face_count

    files = in_report + ["results/figures/val_examples.png"] + \
        sorted(str(p.relative_to(ROOT)) for p in (ROOT / "results" / "failures").glob("*.png")) + \
        sorted(str(p.relative_to(ROOT)) for p in (ROOT / "results" / "leak_test_val").glob("*.jpg"))
    rows = []
    for f in dict.fromkeys(files):
        img = cv2.cvtColor(cv2.imread(str(ROOT / f)), cv2.COLOR_BGR2RGB)
        rows.append({"file": f, "in_report": f in in_report or f == "results/figures/val_examples.png",
                     "shows_pixels": True, "faces_still_detected_after_redaction": face_count(img),
                     "checked_by_yash": "", "plates_or_faces_to_add_to_redact_csv": ""})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "redaction_checklist.csv", index=False)
    return df


def main() -> None:
    cfg = load_config()
    figs = {"confusion_irc": confusion_irc(cfg), "failures_first5": first_five_failures(cfg)}
    check = redaction_checklist(cfg, [figs["failures_first5"]])
    write_result("report_figures", {"figures": figs, "redaction_checklist": "results/redaction_checklist.csv"}, cfg)
    print(figs)
    print(check.to_string())


if __name__ == "__main__":
    main()
