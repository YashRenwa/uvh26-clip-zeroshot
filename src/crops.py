"""One square-context crop per ST box, plus the crop manifest and its tags.

Crop recipe (identical for both arms): square around the box centre with side = context x max(w, h),
clipped to the frame, any out-of-frame part padded with CLIP's mean pixel. CLIPProcessor then resizes
and centre-crops it unchanged; because the crop is already square, the vehicle is never cut off.

Outputs:
    results/crops_{val,test}.csv   crop_id, stem, split, group, cls, class, box, long side, size bin,
                                   IR flag, frame brightness, max IoU with another GT box
    results/data_summary.json      box counts, dedup, class x split table, check against Assignment 1
    results/split_class_counts.csv
    results/figures/val_examples.png  one seeded random val crop per class, redacted

    python -m src.crops [--smoke]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import ROOT, add_common_args, load_config, read_split, set_seed, write_result
from src.data import group_of, load_boxes, read_frame

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)   # CLIPImageProcessor.image_mean
FILL = tuple(int(round(255 * m)) for m in CLIP_MEAN)
A1_SPLIT_JSON = ROOT.parent / "uvh26-pcu-counting" / "results" / "split.json"


def square_window(x1: float, y1: float, x2: float, y2: float, context: float) -> tuple[int, int, int]:
    """(left, top, side) of the square crop window in frame pixels (may extend past the frame)."""
    side = max(1, int(round(context * max(x2 - x1, y2 - y1))))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def square_crop(frame: np.ndarray, box: tuple[float, float, float, float], context: float) -> np.ndarray:
    """Square-context crop of `frame` (RGB uint8); out-of-frame pixels are CLIP's mean pixel."""
    left, top, side = square_window(*box, context)
    H, W = frame.shape[:2]
    out = np.empty((side, side, 3), np.uint8)
    out[:] = FILL
    fx1, fy1, fx2, fy2 = max(left, 0), max(top, 0), min(left + side, W), min(top + side, H)
    if fx2 > fx1 and fy2 > fy1:
        out[fy1 - top:fy2 - top, fx1 - left:fx2 - left] = frame[fy1:fy2, fx1:fx2]
    return out


def max_iou_with_others(boxes: np.ndarray) -> np.ndarray:
    """For each box (x1, y1, x2, y2), its largest IoU with any other box in the frame (occlusion proxy)."""
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    x1, y1, x2, y2 = (boxes[:, i] for i in range(4))
    iw = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = iw * ih
    area = (x2 - x1) * (y2 - y1)
    iou = inter / (area[:, None] + area - inter)
    np.fill_diagonal(iou, 0)
    return iou.max(axis=1)


def frame_tags(frame: np.ndarray) -> tuple[float, float]:
    """(colourfulness = mean |R-G| + |G-B|, brightness = mean grey level 0-255)."""
    f = frame.astype(np.int16)
    colour = float(np.abs(f[..., 0] - f[..., 1]).mean() + np.abs(f[..., 1] - f[..., 2]).mean())
    return colour, float(frame.mean())


def size_bin(long_side: float, edges: list[int]) -> str:
    lo, hi = edges
    return f"<{lo}" if long_side < lo else f"{lo}-{hi - 1}" if long_side < hi else f">={hi}"


def build_manifest(cfg: dict[str, Any], split: str, stems: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Crop manifest for one split (no pixels stored), plus the box-count summary."""
    boxes, summary = load_boxes(cfg, stems)

    def tags(stem: str) -> tuple[str, float, float]:
        return (stem, *frame_tags(read_frame(cfg, stem)))

    with ThreadPoolExecutor(max_workers=8) as ex:
        ft = {s: (c, b) for s, c, b in tqdm(ex.map(tags, stems), total=len(stems), desc=f"{split} frames")}
    parts = []
    for stem, df in boxes.groupby("stem", sort=False):
        df = df.copy()
        df["max_iou"] = max_iou_with_others(df[["x1", "y1", "x2", "y2"]].to_numpy())
        parts.append(df)
    m = pd.concat(parts, ignore_index=True)
    m.insert(0, "crop_id", m["stem"] + "_" + m["box_idx"].map("{:03d}".format))
    m.insert(2, "split", split)
    m.insert(3, "group", m["stem"].map(lambda s: group_of(cfg, s)))
    m["box_w"], m["box_h"] = m["x2"] - m["x1"], m["y2"] - m["y1"]
    m["long_side"] = np.maximum(m["box_w"], m["box_h"])
    m["size_bin"] = m["long_side"].map(lambda v: size_bin(v, cfg["crops"]["size_bins"]))
    m["colourfulness"] = m["stem"].map(lambda s: ft[s][0])
    m["ir"] = m["colourfulness"] < cfg["crops"]["ir_threshold"]
    m["brightness"] = m["stem"].map(lambda s: ft[s][1])
    win = [square_window(*b, cfg["crops"]["context"]) for b in m[["x1", "y1", "x2", "y2"]].to_numpy()]
    m["crop_left"], m["crop_top"], m["crop_side"] = zip(*win)
    m["truncated"] = (m["x1"] <= 0) | (m["y1"] <= 0) | (m["x2"] >= m["img_w"]) | (m["y2"] >= m["img_h"])
    m = m.drop(columns=["box_idx"])
    summary.update(ir_images=int(sum(c < cfg["crops"]["ir_threshold"] for c, _ in ft.values())))
    return m.round(3), summary


def iter_crops(cfg: dict[str, Any], manifest: pd.DataFrame) -> Iterator[tuple[str, np.ndarray]]:
    """(crop_id, crop RGB uint8) in manifest order; each frame is read once. Nothing is written to disk."""
    context = cfg["crops"]["context"]
    for stem, df in manifest.groupby("stem", sort=False):
        frame = read_frame(cfg, stem)
        for r in df.itertuples(index=False):
            yield r.crop_id, square_crop(frame, (r.x1, r.y1, r.x2, r.y2), context)


def example_grid(cfg: dict[str, Any], manifest: pd.DataFrame, out: Path) -> Path:
    """One seeded random val crop per class, redacted, with class name and support."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.privacy import redact

    rng = np.random.default_rng(cfg["seed"])
    classes = cfg["dataset"]["classes"]
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.6))
    for ax, name in zip(axes.ravel(), classes):
        ax.axis("off")
        rows = manifest[manifest["class"] == name]
        if rows.empty:
            ax.set_title(f"{name}\n(none)", fontsize=9)
            continue
        r = rows.iloc[int(rng.integers(len(rows)))]
        frame = read_frame(cfg, r["stem"])
        crop = square_crop(frame, (r.x1, r.y1, r.x2, r.y2), cfg["crops"]["context"])
        crop = redact(crop, r["stem"], origin=(r.crop_left, r.crop_top))
        ax.imshow(crop)
        ax.set_title(f"{name}\n{int(r.long_side)} px, n={len(rows)}", fontsize=9)
    fig.suptitle("One random val crop per class (seeded); n = val crops of that class", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def a1_reference() -> dict[str, dict[str, int]] | None:
    """Assignment 1's ST class counts per split (after dedup), if its results are on disk."""
    if not A1_SPLIT_JSON.exists():
        return None
    cc = json.loads(A1_SPLIT_JSON.read_text())["class_counts"]
    return {"val": cc["st/val"], "test": cc["st/test"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    set_seed(cfg["seed"])
    res = cfg["paths"]["results_dir"]
    classes = cfg["dataset"]["classes"]

    splits = {"val": read_split(cfg, "val"), "test": read_split(cfg, "test")}
    if cfg["smoke"]:
        splits = {"val": splits["val"][: cfg["smoke_images"]]}
    out: dict[str, Any] = {"crop_recipe": {"context": cfg["crops"]["context"], "fill_rgb": FILL,
                                           "size_bins_px": cfg["crops"]["size_bins"],
                                           "ir_threshold": cfg["crops"]["ir_threshold"]}}
    table = {}
    for split, stems in splits.items():
        m, summary = build_manifest(cfg, split, stems)
        res.mkdir(parents=True, exist_ok=True)
        m.to_csv(res / f"crops_{split}.csv", index=False)
        counts = m["class"].value_counts().reindex(classes, fill_value=0)
        table[split] = counts
        summary.update(
            crops=len(m), groups=int(m["group"].nunique()),
            size_bins=m["size_bin"].value_counts().to_dict(),
            ir_crops=int(m["ir"].sum()), truncated_crops=int(m["truncated"].sum()),
            occluded_iou_ge_0_3=int((m["max_iou"] >= 0.3).sum()),
            low_support_classes=[c for c in classes if counts[c] < cfg["dataset"]["low_support_min"]],
        )
        out[split] = summary
        print(f"{split}: {summary['images']} images, {summary['raw_boxes']} raw boxes, "
              f"{summary['exact_duplicates_dropped']} duplicates dropped -> {len(m)} crops")
    t = pd.DataFrame(table)
    t.loc["total"] = t.sum()
    t.to_csv(res / "split_class_counts.csv")
    out["class_counts"] = {s: {c: int(t.loc[c, s]) for c in classes} for s in table}
    ref = a1_reference()
    if ref and not cfg["smoke"]:
        diff = {s: {c: out["class_counts"][s][c] - ref[s][c] for c in classes
                    if out["class_counts"][s][c] != ref[s][c]} for s in table}
        out["matches_assignment1"] = all(not d for d in diff.values())
        out["diff_vs_assignment1"] = diff
        assert out["matches_assignment1"], f"class counts differ from A1: {diff}"
    print(t.to_string())
    out["example_figure"] = str(example_grid(cfg, pd.read_csv(res / "crops_val.csv", dtype={"stem": str}),
                                             res / "figures" / "val_examples.png"))
    write_result("data_summary", out, cfg)
    print(f"wrote {res / 'data_summary.json'}")


if __name__ == "__main__":
    main()
