"""Score one prompt arm on val crops (never test).

    python -m src.run_val --prompts prompts/baseline.yaml [--smoke]
    python -m src.run_val --prompts prompts/ensemble.yaml --draft   # prompt design on val (Phase 3)

Outputs: results/val_<arm>.json (accuracy, balanced accuracy, macro-F1, per-class P/R/F1/support,
IRC accuracy, PCU MAE, timing), results/val_predictions_<arm>.csv, results/val_confusion_<arm>.csv.
Image embeddings go to the gitignored dev cache (data/cache) so later prompt versions reuse them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common import ROOT, add_common_args, load_config, set_seed, write_result
from src.metrics import classification_report
from src.pcu import class_factor_vector, pcu_errors, pcu_factors
from src.zeroshot import (embedding_fingerprint, image_embeddings, load_model, logit_scale, predict,
                          text_classifier)


def val_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    """Val crop manifest (smoke: the first `smoke_images` val images)."""
    m = pd.read_csv(ROOT / "results" / "crops_val.csv", dtype={"stem": str, "group": str})
    if cfg["smoke"]:
        stems = m["stem"].drop_duplicates().head(cfg["smoke_images"])
        m = m[m["stem"].isin(stems)].reset_index(drop=True)
    return m


def low_support_classes() -> list[str]:
    """Classes with fewer than low_support_min TEST crops (fixed by the manifest, same list for every table)."""
    return json.loads((ROOT / "results" / "data_summary.json").read_text())["test"]["low_support_classes"]


def score_arm(cfg, model, processor, manifest, img_emb, prompt_path: Path, draft_ok: bool) -> dict[str, Any]:
    """Every val number for one prompt file, from the shared image embeddings."""
    classes = cfg["dataset"]["classes"]
    cls_emb, info = text_classifier(cfg, model, processor, prompt_path, draft_ok=draft_ok)
    pred, score, sm = predict(img_emb, cls_emb, logit_scale(model))
    true = manifest["cls"].to_numpy()
    low = low_support_classes()
    rep = classification_report(cfg, true, pred, sm, low)
    pcu = pcu_errors(manifest["stem"].to_numpy(), true, pred, class_factor_vector(cfg))
    return {"prompts": info, "logit_scale": logit_scale(model), "image_embeddings_sha256": embedding_fingerprint(img_emb),
            "low_support_classes": low, **rep, "pcu": pcu, "mean_score": float(score.mean()),
            "_pred": pred, "_score": score, "_softmax": sm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--draft", action="store_true", help="allow an unfrozen prompt YAML (val design only)")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(cfg["seed"])
    path = Path(args.prompts).resolve()
    arm = yaml.safe_load(path.read_text())["arm"]
    classes = cfg["dataset"]["classes"]

    manifest = val_manifest(cfg)
    model, processor = load_model(cfg)
    cache = cfg["paths"]["cache_dir"] / f"img_emb_val{'_smoke' if cfg['smoke'] else ''}.npz"
    img_emb, timing = image_embeddings(cfg, model, processor, manifest, cache=cache)
    res = score_arm(cfg, model, processor, manifest, img_emb, path, args.draft)
    pred, score, sm = res.pop("_pred"), res.pop("_score"), res.pop("_softmax")
    out = {"arm": arm, "split": "val", "model": cfg["model"]["id"], "model_revision": cfg["model"]["revision"],
           "seed": cfg["seed"], "pcu_factors": pcu_factors(cfg)["class_factor"], "timing": timing, **res}
    write_result(f"val_{arm}", out, cfg)

    res_dir = cfg["paths"]["results_dir"]
    pd.DataFrame({"crop_id": manifest["crop_id"], "true": manifest["class"],
                  "pred": np.array(classes)[pred], "score": score.round(5)}).to_csv(
        res_dir / f"val_predictions_{arm}.csv", index=False)
    cm = pd.DataFrame(out["confusion"], index=classes, columns=classes)
    cm.to_csv(res_dir / f"val_confusion_{arm}.csv")

    print(f"{arm} on val ({out['n']} crops): accuracy {out['accuracy']:.3f}, balanced {out['balanced_accuracy']:.3f}, "
          f"macro-F1 {out['macro_f1_headline']:.3f} (all present {out['macro_f1_all_present']:.3f}), "
          f"IRC accuracy {out['irc_accuracy']:.3f}, PCU MAE {out['pcu']['pcu_mae']:.2f} "
          f"({100 * out['pcu']['pcu_rel_mae']:.1f}% of mean {out['pcu']['mean_true_pcu']:.2f}), "
          f"bias {out['pcu']['pcu_bias']:+.2f}")
    print(f"timing: {timing}")
    short = [c[:6] for c in classes]
    print("confusion (rows true, cols predicted):")
    print(pd.DataFrame(out["confusion"], index=classes, columns=short).to_string())


if __name__ == "__main__":
    main()
