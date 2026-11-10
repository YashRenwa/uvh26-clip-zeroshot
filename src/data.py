"""UVH-26 frames and STAPLE (ST) boxes for the val and test images.

Files come from UVH_RAW_ROOT when they are there (locally: Assignment 1's data/raw), otherwise they are
downloaded from the Hub at the pinned dataset revision. Nothing is ever uploaded.

MV and ST JSONs use different image ids, so images are joined on file_name. COCO category_id k is
class index k-1. The ST JSONs contain exact duplicate boxes; they are dropped with Assignment 1's rule
(identical YOLO label line after clipping to the frame), so counts and PCU match A1.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def raw_file(cfg: dict[str, Any], rel: str) -> Path:
    """Local path of a dataset file; downloads it at the pinned revision if it is not under raw_root."""
    path = cfg["paths"]["raw_root"] / rel
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(cfg["dataset"]["repo_id"], rel, repo_type="dataset",
                                revision=cfg["dataset"]["revision"], local_dir=cfg["paths"]["raw_root"]))


@lru_cache(maxsize=1)
def _pool(splits_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(splits_dir) / "pool.csv", dtype=str).set_index("stem")


@lru_cache(maxsize=1)
def _groups(splits_dir: str) -> dict[str, str]:
    with open(Path(splits_dir) / "groups.csv") as f:
        return {r["stem"]: r["group"] for r in csv.DictReader(f)}


def pool(cfg: dict[str, Any]) -> pd.DataFrame:
    """splits/pool.csv indexed by stem (file_name, official_split, repo_path)."""
    return _pool(str(cfg["paths"]["splits_dir"]))


def group_of(cfg: dict[str, Any], stem: str) -> str:
    """Pseudo-camera group id from Assignment 1."""
    return _groups(str(cfg["paths"]["splits_dir"]))[stem]


def read_frame(cfg: dict[str, Any], stem: str) -> np.ndarray:
    """Original 1920x1080 frame as an RGB uint8 array."""
    path = raw_file(cfg, pool(cfg).loc[stem, "repo_path"])
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _a1_line(cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    """Assignment 1's YOLO label line; identical lines are exact duplicates."""
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def load_boxes(cfg: dict[str, Any], stems: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """ST boxes for `stems`, clipped to the frame, exact duplicates dropped.

    Returns one row per kept box (stem, box_idx, cls, class, x1, y1, x2, y2, img_w, img_h) and a
    summary with raw / kept / dropped counts.
    """
    wanted = {pool(cfg).loc[s, "file_name"]: s for s in stems}
    classes = cfg["dataset"]["classes"]
    rows, raw, degenerate, dup = [], 0, 0, 0
    seen_files = set()
    for rel in cfg["dataset"]["annotations"].values():
        with open(raw_file(cfg, rel)) as f:
            coco = json.load(f)
        ims = {im["id"]: im for im in coco["images"] if im["file_name"] in wanted}
        seen_files |= {im["file_name"] for im in ims.values()}
        per_image: dict[int, list] = {i: [] for i in ims}
        for a in coco["annotations"]:
            if a["image_id"] in per_image:
                per_image[a["image_id"]].append(a)
        for iid, anns in per_image.items():
            im = ims[iid]
            w, h, stem = im["width"], im["height"], wanted[im["file_name"]]
            lines: set[str] = set()
            for a in anns:
                raw += 1
                cls = a["category_id"] - 1
                x, y, bw, bh = a["bbox"]
                x1, y1, x2, y2 = max(x, 0.0), max(y, 0.0), min(x + bw, w), min(y + bh, h)
                if x2 <= x1 or y2 <= y1:
                    degenerate += 1
                    continue
                line = _a1_line(cls, x1, y1, x2, y2, w, h)
                if line in lines:
                    dup += 1
                    continue
                lines.add(line)
                rows.append((stem, cls, classes[cls], x1, y1, x2, y2, w, h))
    missing = set(wanted) - seen_files
    if missing:
        raise RuntimeError(f"{len(missing)} images have no ST entry, e.g. {sorted(missing)[:3]}")
    df = pd.DataFrame(rows, columns=["stem", "cls", "class", "x1", "y1", "x2", "y2", "img_w", "img_h"])
    df["box_idx"] = df.groupby("stem").cumcount()
    summary = {"images": len(stems), "raw_boxes": raw, "degenerate_dropped": degenerate,
               "exact_duplicates_dropped": dup, "boxes": len(df)}
    return df, summary
