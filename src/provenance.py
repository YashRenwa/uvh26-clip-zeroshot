"""Split checksum, pinned revisions and contamination facts -> results/provenance.json.

    python -m src.provenance [--smoke] [--offline]
"""

from __future__ import annotations

import argparse
import csv

from src.common import add_common_args, load_config, read_split, run_header, verify_test_split, write_result

# Facts checked by hand on 2026-11-10 against the pages listed; short quotes are the evidence.
CONTAMINATION = {
    "uvh26": {
        "title": "The Urban Vision Hackathon Dataset and Models: Towards Image Annotations and "
                 "Accurate Vision Models for Indian Traffic",
        "arxiv": "2511.02563",
        "first_public": "2025-11-04 (arXiv v1)",
        "frames_captured": "February 2025, a 25-day period, 06:00-18:00 IST",
        "evidence": "\"a 25-day period in February, 2025\" (section 2.1)",
        "source": ["https://arxiv.org/abs/2511.02563", "https://arxiv.org/html/2511.02563"],
        "cameras": "about 2,800 Bengaluru Safe-City CCTV cameras",
        "license": "CC BY 4.0 (dataset)",
    },
    "clip": {
        "model": "openai/clip-vit-base-patch16",
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "arxiv": "2103.00020",
        "model_date": "January 2021 (model card); paper v1 2021-02-26",
        "training_data": "WIT (WebImageText), 400M image-text pairs collected from the internet",
        "source": ["https://arxiv.org/abs/2103.00020", "https://arxiv.org/html/2103.00020v1",
                   "https://huggingface.co/openai/clip-vit-base-patch16"],
    },
    "conclusion": (
        "UVH-26 frames were captured in February 2025, four years after CLIP's training data was "
        "collected (model dated January 2021), so no test frame can be in CLIP's training set."
    ),
    "caveat": (
        "The concepts (auto-rickshaws, Indian buses, CCTV-style images) were certainly in CLIP's web "
        "data, so this tests transfer to a new camera domain, not unseen classes."
    ),
    "own_leakage": (
        "Prompts and the review threshold are chosen on val cameras only. Test pseudo-cameras are "
        "disjoint from val by construction; residual same-camera leakage from Assignment 1's audit is "
        "17.5% (95% CI 8.7-32%, 40 random test images vs their most similar train image). The audit "
        "measured test vs train; nothing is trained here, so it bounds val-to-test camera overlap only "
        "loosely."
    ),
    "a1_clip_use": (
        "Assignment 1 used CLIP ViT-B/32 embeddings only to compare grouping rules; the frozen split "
        "used 64x36 gradient-magnitude correlation, so no CLIP signal shaped the test set."
    ),
}


def group_overlap(cfg) -> dict:
    """Pseudo-camera groups shared by val and test (expected 0)."""
    with open(cfg["paths"]["splits_dir"] / "groups.csv") as f:
        group = {r["stem"]: r["group"] for r in csv.DictReader(f)}
    val, test = read_split(cfg, "val"), read_split(cfg, "test")
    gv, gt = {group[s] for s in val}, {group[s] for s in test}
    return {"val_images": len(val), "val_groups": len(gv), "test_images": len(test),
            "test_groups": len(gt), "groups_shared_val_test": len(gv & gt),
            "images_shared_val_test": len(set(val) & set(test))}


def hub_revisions(cfg) -> dict:
    """Confirm the pinned revisions exist on the Hub and record the dataset licence tag."""
    from huggingface_hub import HfApi

    api = HfApi()
    m = api.model_info(cfg["model"]["id"], revision=cfg["model"]["revision"])
    d = api.dataset_info(cfg["dataset"]["repo_id"], revision=cfg["dataset"]["revision"])
    assert m.sha == cfg["model"]["revision"], (m.sha, cfg["model"]["revision"])
    assert d.sha == cfg["dataset"]["revision"], (d.sha, cfg["dataset"]["revision"])
    return {"model_sha": m.sha, "model_last_modified": m.last_modified,
            "dataset_sha": d.sha, "dataset_last_modified": d.last_modified,
            "dataset_license_tag": (d.card_data or {}).get("license")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--offline", action="store_true", help="skip the Hub revision check")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    out = {
        "test_sha256": verify_test_split(cfg),
        "split": group_overlap(cfg),
        "split_source": "../uvh26-pcu-counting/splits at commit ef77f21 (Assignment 1)",
        "run": run_header(cfg),
        "hub": None if args.offline else hub_revisions(cfg),
        "contamination": CONTAMINATION,
    }
    assert out["split"]["groups_shared_val_test"] == 0 and out["split"]["images_shared_val_test"] == 0
    path = write_result("provenance", out, cfg)
    print(f"test.txt sha256 OK: {out['test_sha256']}")
    print(f"split: {out['split']}")
    print(f"model {cfg['model']['id']} @ {cfg['model']['revision']}")
    print(f"dataset {cfg['dataset']['repo_id']} @ {cfg['dataset']['revision']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
