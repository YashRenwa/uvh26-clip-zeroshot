"""Failure analysis on test for the headline arm (Phase 6).

Draws are random with the project seed, never picked by hand:
  1. n_random 14-class errors (numpy default_rng(seed).choice over all errors, without replacement);
     rank = draw order, so the report's five examples are ranks 1-5.
  2. n_confident errors that pass the review rule unseen: IRC score >= frozen t and IRC category wrong
     (the errors that reach PCU), drawn with the same generator, excluding crops already drawn.

Outputs:
    results/failures/{random,confident}_sheet_N.png   10 errors per sheet, redacted
    analysis/failure_tags.csv      tag template; suggested_cause is a SUGGESTION from the tags, the
                                   final 'cause' is Yash's (G3)
    results/failure_patterns.json, results/failure_patterns_*.csv   error rates over ALL test crops

    python -m src.failures [--smoke]
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.common import ROOT, add_common_args, load_config, write_result
from src.crops import square_crop
from src.data import read_frame
from src.privacy import redact

CAUSES = ["lookalike", "occlusion", "small", "ir_night", "truncated", "context", "label", "other"]
# Fine-grained pairs the plan names as lookalikes, beyond classes sharing an IRC category.
LOOKALIKE_PAIRS = {frozenset(p) for p in [("Tempo-traveller", "Mini-bus"), ("Truck", "LCV"), ("Bus", "Mini-bus"),
                                           ("Van", "Tempo-traveller"), ("Van", "MUV"), ("LCV", "Three-wheeler")]}


def load_test(cfg: dict[str, Any], arm: str, split: str) -> pd.DataFrame:
    res = cfg["paths"]["results_dir"]
    pred = pd.read_csv(res / f"{split}_predictions.csv")
    man = pd.read_csv(ROOT / "results" / ("crops_test.csv" if split == "test" else "crops_val.csv"),
                      dtype={"stem": str, "group": str})
    df = man.merge(pred, on="crop_id", validate="one_to_one")
    cmap = cfg["pcu"]["class_map"]
    df["pred"], df["score"] = df[f"{arm}_pred"], df[f"{arm}_score"]
    df["irc_pred"], df["irc_score"] = df[f"{arm}_irc_pred"], df[f"{arm}_irc_score"]
    df["irc_true"] = df["class"].map(cmap)
    df["wrong"] = df["pred"] != df["class"]
    df["irc_wrong"] = df["irc_pred"] != df["irc_true"]
    return df


def suggest_cause(r, cfg: dict[str, Any]) -> str:
    """A SUGGESTION from the crop's tags, in a fixed order; Yash assigns the final cause."""
    if r["truncated"]:
        return "truncated"
    if r["ir"]:
        return "ir_night"
    if r["size_bin"] != f">={cfg['crops']['size_bins'][1]}":
        return "small"
    if r["max_iou"] >= cfg["failures"]["occlusion_iou"]:
        return "occlusion"
    cmap = cfg["pcu"]["class_map"]
    if cmap[r["class"]] == cmap[r["pred"]] or frozenset((r["class"], r["pred"])) in LOOKALIKE_PAIRS:
        return "lookalike"
    return "other"


def draw(cfg: dict[str, Any], df: pd.DataFrame, t: float) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    err = df.index[df["wrong"]].to_numpy()
    n_r = min(cfg["failures"]["n_random"], len(err))
    pick_r = rng.choice(err, n_r, replace=False)
    conf_pool = df.index[(df["irc_score"] >= t) & df["irc_wrong"]].to_numpy()
    conf_pool = np.setdiff1d(conf_pool, pick_r)
    n_c = min(cfg["failures"]["n_confident"], len(conf_pool))
    pick_c = rng.choice(conf_pool, n_c, replace=False)
    a = df.loc[pick_r].assign(set="random", rank=np.arange(1, n_r + 1))
    b = df.loc[pick_c].assign(set="confident", rank=np.arange(1, n_c + 1))
    return pd.concat([a, b])


def render_sheet(cfg: dict[str, Any], rows: pd.DataFrame, title: str, out) -> None:
    """One error per row: redacted crop | redacted frame with the box | tags."""
    side, fw, fh, tw, gap = 260, 462, 260, 470, 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    blocks = []
    for d in rows.to_dict("records"):
        r = SimpleNamespace(**{k: v for k, v in d.items() if k != "class"}, true_class=d["class"])
        frame = redact(read_frame(cfg, r.stem), r.stem)
        crop = square_crop(frame, (r.x1, r.y1, r.x2, r.y2), cfg["crops"]["context"])
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA if crop.shape[0] > side else cv2.INTER_CUBIC)
        sx, sy = fw / frame.shape[1], fh / frame.shape[0]
        thumb = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (int(r.x1 * sx), int(r.y1 * sy)), (int(r.x2 * sx), int(r.y2 * sy)), (235, 104, 52), 2)
        block = np.full((side, side + fw + tw + 2 * gap, 3), 255, np.uint8)
        block[:, :side], block[:fh, side + gap:side + gap + fw] = crop, thumb
        lines = [f"#{r.rank} ({r.set})   {r.crop_id}",
                 f"true: {r.true_class}  [{r.irc_true}]",
                 f"pred: {r.pred}  [{r.irc_pred}]",
                 f"score {r.score:.3f}   IRC score {r.irc_score:.3f}",
                 f"size {int(r.long_side)} px ({r.size_bin})   IR {'yes' if r.ir else 'no'}",
                 f"max IoU other box {r.max_iou:.2f}   edge {'yes' if r.truncated else 'no'}",
                 f"suggested (not a tag): {r.suggested_cause}"]
        x0 = side + fw + 2 * gap + 6
        for i, text in enumerate(lines):
            cv2.putText(block, text, (x0, 30 + 32 * i), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        blocks.append(cv2.cvtColor(block, cv2.COLOR_RGB2BGR))
        blocks.append(np.full((gap, block.shape[1], 3), 255, np.uint8))
    head = np.full((44, blocks[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (6, 30), font, 0.7, (11, 11, 11), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, *blocks]))


def rate_table(df: pd.DataFrame, by: str, order: list | None = None) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({"crops": g.size(), "class_errors": g["wrong"].sum(), "irc_errors": g["irc_wrong"].sum(),
                      "accepted": g["accepted"].sum(), "accepted_irc_errors": g["accepted_irc_wrong"].sum()})
    t["class_error_rate"] = (t["class_errors"] / t["crops"]).round(4)
    t["irc_error_rate"] = (t["irc_errors"] / t["crops"]).round(4)
    t["share_accepted"] = (t["accepted"] / t["crops"]).round(4)
    t["share_of_all_irc_errors"] = (t["irc_errors"] / df["irc_wrong"].sum()).round(4)
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    return t


def patterns(cfg: dict[str, Any], df: pd.DataFrame, t: float) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df["accepted"] = df["irc_score"] >= t
    df["accepted_irc_wrong"] = df["accepted"] & df["irc_wrong"]
    lo, hi = cfg["crops"]["size_bins"]
    edges = cfg["failures"]["iou_bins"]
    df["occlusion_bin"] = np.select(
        [df["max_iou"] <= edges[0], df["max_iou"] < edges[1], df["max_iou"] < edges[2]],
        ["0", f"(0, {edges[1]})", f"[{edges[1]}, {edges[2]})"], f">={edges[2]}")
    df["ir_frame"] = np.where(df["ir"], "IR / greyscale", "colour")
    df["at_frame_edge"] = np.where(df["truncated"], "touches edge", "inside frame")
    return {
        "size_bin": rate_table(df, "size_bin", [f"<{lo}", f"{lo}-{hi - 1}", f">={hi}"]),
        "ir": rate_table(df, "ir_frame", ["colour", "IR / greyscale"]),
        "occlusion_iou": rate_table(df, "occlusion_bin", ["0", f"(0, {edges[1]})", f"[{edges[1]}, {edges[2]})", f">={edges[2]}"]),
        "frame_edge": rate_table(df, "at_frame_edge", ["inside frame", "touches edge"]),
        "true_class": rate_table(df, "class", cfg["dataset"]["classes"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    rule = json.loads((ROOT / "results" / "frozen_rule.json").read_text())
    arm = rule["headline_arm"]
    t = rule["rules"][arm]["threshold"]
    split = "val_smoke" if args.smoke else "test"
    df = load_test(cfg, arm, split)
    df["suggested_cause"] = [suggest_cause(r, cfg) for r in df.to_dict("records")]
    picks = draw(cfg, df, t)

    res = cfg["paths"]["results_dir"]
    sheets = []
    for s in ("random", "confident"):
        rows = picks[picks["set"] == s]
        for k in range(0, len(rows), 10):
            p = res / "failures" / f"{s}_sheet_{k // 10 + 1}.png"
            what = ("random 14-class errors" if s == "random" else "errors passing the review rule (IRC score >= t, IRC wrong)")
            render_sheet(cfg, rows.iloc[k:k + 10], f"{arm} on {split}: {what}, seed {cfg['seed']}, ranks {k + 1}-{k + len(rows.iloc[k:k + 10])}", p)
            sheets.append(str(p.relative_to(res)))

    tags = picks[["set", "rank", "crop_id", "stem", "class", "pred", "score", "irc_true", "irc_pred", "irc_score",
                  "size_bin", "long_side", "ir", "max_iou", "truncated", "suggested_cause"]].rename(
        columns={"class": "true", "suggested_cause": "suggested_cause_NOT_A_TAG"})
    tags["cause"] = ""
    tags["note"] = ""
    tag_path = (ROOT / "analysis" / "failure_tags.csv") if not args.smoke else res / "failure_tags.csv"
    if tag_path.exists() and not args.smoke and pd.read_csv(tag_path)["cause"].notna().any():
        raise SystemExit(f"{tag_path} already has tags; not overwriting")
    tag_path.parent.mkdir(parents=True, exist_ok=True)
    tags.to_csv(tag_path, index=False)

    tables = patterns(cfg, df, t)
    for name, tab in tables.items():
        tab.to_csv(res / f"failure_patterns_{name}.csv")
    accepted = df["irc_score"] >= t
    out = {
        "arm": arm, "split": split, "threshold": t, "seed": cfg["seed"], "crops": len(df),
        "class_errors": int(df["wrong"].sum()), "irc_errors": int(df["irc_wrong"].sum()),
        "accepted": int(accepted.sum()),
        "accepted_class_wrong": int((accepted & df["wrong"]).sum()),
        "accepted_irc_wrong": int((accepted & df["irc_wrong"]).sum()),
        "accepted_class_wrong_irc_right": int((accepted & df["wrong"] & ~df["irc_wrong"]).sum()),
        "draw": {"random": {"pool": int(df["wrong"].sum()), "n": int((picks["set"] == "random").sum())},
                 "confident": {"pool_definition": "IRC score >= t and IRC category wrong, excluding the random draw",
                               "pool": int((accepted & df["irc_wrong"]).sum()), "n": int((picks["set"] == "confident").sum())},
                 "method": "numpy default_rng(seed).choice without replacement; random draw first, then confident"},
        "suggested_cause_counts_NOT_TAGS": picks.groupby("set")["suggested_cause"].value_counts().unstack(fill_value=0).to_dict("index"),
        "sheets": sheets, "tag_template": str(tag_path.relative_to(ROOT)),
        "patterns": {k: v.reset_index().to_dict("records") for k, v in tables.items()},
    }
    write_result("failure_patterns", out, cfg)
    print({k: out[k] for k in ("crops", "class_errors", "irc_errors", "accepted", "accepted_irc_wrong",
                               "accepted_class_wrong_irc_right")})
    for k, v in tables.items():
        print(f"\n{k}\n{v.to_string()}")


if __name__ == "__main__":
    main()
