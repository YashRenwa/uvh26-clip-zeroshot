"""Prompt-ensemble design on val (Phase 3): score one draft version and append it to the log.

    python -m src.prompt_design --version v1 "reason for this version" [--smoke]

Reads prompts/ensemble_<version>.yaml as a val-only draft, reuses the shared val image embeddings, and
writes results/prompt_log.csv (one row per version, baseline row first) and
results/prompt_versions/<version>_confusion{,_irc}.csv. At most 3 versions are allowed.
Selection rule (Yash, 2026-11-10): best val IRC accuracy, PCU MAE as the tiebreak.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from src.common import ROOT, add_common_args, load_config, set_seed
from src.pcu import categories
from src.run_val import score_arm, val_manifest
from src.zeroshot import image_embeddings, load_model

MAX_VERSIONS = 3
DISCLOSURE = "baseline val confusion known when v1 was written"
TRACKED = ["Three-wheeler", "LCV", "Others", "Hatchback"]


def log_row(cfg, version: str, path, res: dict, reason: str) -> dict:
    classes = cfg["dataset"]["classes"]
    pred_counts = [res["per_class"][c]["predicted"] for c in classes]
    row = {"version": version, "file": str(path.relative_to(ROOT)),
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": res["n"],
           "accuracy": res["accuracy"], "balanced_accuracy": res["balanced_accuracy"],
           "macro_f1_headline": res["macro_f1_headline"], "irc_accuracy": res["irc_accuracy"],
           "pcu_mae": res["pcu"]["pcu_mae"], "pcu_bias": res["pcu"]["pcu_bias"],
           "share_pred_others": pred_counts[classes.index("Others")] / res["n"]}
    row |= {f"recall_{c}": res["per_class"][c]["recall"] for c in TRACKED}
    row |= {"disclosure": DISCLOSURE if version != "baseline" else "", "reason": reason}
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--version", required=True, help="v1, v2 or v3 (or 'baseline' for the reference row)")
    ap.add_argument("reason")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(cfg["seed"])
    res_dir = cfg["paths"]["results_dir"]
    log_path = res_dir / "prompt_log.csv"
    log = pd.read_csv(log_path, dtype=str) if log_path.exists() else pd.DataFrame()
    drafts = [v for v in log.get("version", pd.Series(dtype=str)) if v != "baseline"]
    if args.version != "baseline" and args.version not in drafts and len(drafts) >= MAX_VERSIONS:
        raise SystemExit(f"already {len(drafts)} versions logged; the plan allows {MAX_VERSIONS}")

    path = (ROOT / "prompts" / ("baseline.yaml" if args.version == "baseline" else f"ensemble_{args.version}.yaml"))
    manifest = val_manifest(cfg)
    model, processor = load_model(cfg)
    cache = cfg["paths"]["cache_dir"] / f"img_emb_val{'_smoke' if cfg['smoke'] else ''}.npz"
    img_emb, _ = image_embeddings(cfg, model, processor, manifest, cache=cache)
    res = score_arm(cfg, model, processor, manifest, img_emb, path, draft_ok=args.version != "baseline")
    for k in ("_pred", "_score", "_softmax"):
        res.pop(k)

    out = res_dir / "prompt_versions"
    out.mkdir(parents=True, exist_ok=True)
    classes = cfg["dataset"]["classes"]
    pd.DataFrame(res["confusion"], index=classes, columns=classes).to_csv(out / f"{args.version}_confusion.csv")
    cats = categories(cfg)
    pd.DataFrame(res["confusion_irc"], index=cats, columns=cats).to_csv(out / f"{args.version}_confusion_irc.csv")
    (out / f"{args.version}.json").write_text(json.dumps(res, indent=2, default=str))

    row = log_row(cfg, args.version, path, res, args.reason)
    if not log.empty:
        log = log[log["version"] != args.version]
    log = pd.concat([log, pd.DataFrame([row]).astype(str)], ignore_index=True)
    order = ["baseline"] + sorted(v for v in log["version"] if v != "baseline")
    log = log.set_index("version").loc[[v for v in order if v in set(log["version"])]].reset_index()
    log.to_csv(log_path, index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))
    print("IRC confusion (rows true, cols predicted):")
    print(pd.DataFrame(res["confusion_irc"], index=cats, columns=cats).to_string())
    short = [c[:6] for c in classes]
    print(pd.DataFrame(res["confusion"], index=classes, columns=short).to_string())


if __name__ == "__main__":
    main()
