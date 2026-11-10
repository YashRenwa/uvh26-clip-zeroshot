"""Test-vs-val camera leakage, measured with Assignment 1's audit protocol.

Each image becomes a 64x36 Sobel gradient-magnitude thumbnail, centred and unit-norm, so a dot product
is the Pearson correlation g (A1 src/cluster.py::thumbnail_features). A seeded random draw of test
images is paired with each one's most similar val image by g. Yash judges every pair by eye (same
camera / unclear / different); record_verdicts() turns the verdicts into a leakage estimate with a
95% Wilson interval and stores it in results/provenance.json.

Outputs in results/leak_test_val/:
    pairs.csv            rank, test_stem, val_stem, g, test/val group, verdict (blank for Yash)
    g_nearest_val.csv    every test image's nearest val image and g (distribution context)
    sheet_{1..4}.jpg     the pairs, 10 per sheet, redacted
    summary.json

    python -m src.leak [--smoke]          draw and render
    python -m src.leak --record           read verdicts from pairs.csv and record the estimate
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import ROOT, add_common_args, load_config, read_split, write_result
from src.data import group_of, read_frame
from src.privacy import redact

A1_THUMBS = ROOT.parent / "uvh26-pcu-counting" / "data" / "prepared" / "thumbnails.npz"
VERDICTS = ("same", "unclear", "different")


def gradient_thumbnail(frame_rgb: np.ndarray) -> np.ndarray:
    """A1's Tg: grey 64x36 (INTER_AREA), Sobel gradient magnitude, centred and unit-norm."""
    g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    t = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = np.hypot(cv2.Sobel(t, cv2.CV_32F, 1, 0), cv2.Sobel(t, cv2.CV_32F, 0, 1)).ravel()
    m = m - m.mean()
    return m / (np.linalg.norm(m) + 1e-8)


def thumbnails(cfg: dict[str, Any], stems: list[str]) -> np.ndarray:
    """Gradient thumbnails for `stems`, cached in the gitignored dev cache."""
    cache = cfg["paths"]["cache_dir"] / "grad_thumbs.npz"
    if cache.exists():
        z = np.load(cache)
        have = dict(zip(z["stems"].tolist(), z["Tg"]))
        if all(s in have for s in stems):
            return np.stack([have[s] for s in stems])
    with ThreadPoolExecutor(max_workers=8) as ex:
        Tg = np.stack(list(tqdm(ex.map(lambda s: gradient_thumbnail(read_frame(cfg, s)), stems),
                                total=len(stems), desc="thumbnails")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, Tg=Tg, stems=np.array(stems))
    return Tg


def a1_agreement(stems: list[str], Tg: np.ndarray) -> dict[str, float] | None:
    """Correlation of our thumbnails (from PNGs) with A1's (from 1280 px JPEGs), same images."""
    if not A1_THUMBS.exists():
        return None
    z = np.load(A1_THUMBS)
    idx = {s: i for i, s in enumerate(z["stems"].tolist())}
    common = [k for k, s in enumerate(stems) if s in idx]
    r = np.einsum("ij,ij->i", Tg[common], z["Tg"][[idx[stems[k]] for k in common]])
    return {"images": len(common), "min_r": round(float(r.min()), 4), "median_r": round(float(np.median(r)), 4)}


def pair_sheet(cfg: dict[str, Any], pairs: pd.DataFrame, title: str, out, w: int = 640, h: int = 360) -> None:
    """One pair per row: test frame | val frame, both redacted, with rank, g and stems."""
    cap, gap, font = 30, 8, cv2.FONT_HERSHEY_SIMPLEX

    def thumb(stem: str) -> np.ndarray:
        f = redact(read_frame(cfg, stem), stem)
        return cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)

    rows = []
    for r in pairs.itertuples(index=False):
        tile = np.full((h + cap, 2 * w + gap, 3), 255, np.uint8)
        tile[cap:, :w], tile[cap:, w + gap:] = thumb(r.test_stem), thumb(r.val_stem)
        text = f"#{r.rank}   g {r.g:.3f}   TEST {r.test_stem}  |  VAL {r.val_stem}"
        cv2.putText(tile, text, (6, 21), font, 0.6, (11, 11, 11), 1, cv2.LINE_AA)
        rows.append(np.vstack([tile, np.full((gap, tile.shape[1], 3), 255, np.uint8)]))
    head = np.full((44, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (6, 30), font, 0.7, (11, 11, 11), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def draw(cfg: dict[str, Any], n: int = 40, per_sheet: int = 10) -> dict[str, Any]:
    out = cfg["paths"]["results_dir"] / "leak_test_val"
    val, test = read_split(cfg, "val"), read_split(cfg, "test")
    if cfg["smoke"]:
        val, test, n = val[:20], test[:10], 5
    Tv, Tt = thumbnails(cfg, val), thumbnails(cfg, test)
    G = Tt @ Tv.T                                  # g for every (test, val) pair
    best = G.argmax(axis=1)
    near = pd.DataFrame({"test_stem": test, "val_stem": [val[j] for j in best],
                         "g": G[np.arange(len(test)), best].round(4)})
    out.mkdir(parents=True, exist_ok=True)
    near.sort_values("g", ascending=False).to_csv(out / "g_nearest_val.csv", index=False)

    pick = np.random.default_rng(cfg["seed"]).permutation(len(test))[:n]
    pairs = near.iloc[pick].sort_values("g", ascending=False, kind="stable").reset_index(drop=True)
    pairs.insert(0, "rank", np.arange(1, len(pairs) + 1))
    pairs["test_group"] = pairs["test_stem"].map(lambda s: group_of(cfg, s))
    pairs["val_group"] = pairs["val_stem"].map(lambda s: group_of(cfg, s))
    assert (pairs["test_group"] != pairs["val_group"]).all()
    pairs["verdict"] = ""
    pairs["note"] = ""
    pairs.to_csv(out / "pairs.csv", index=False)
    sheets = []
    for k in range(0, len(pairs), per_sheet):
        p = out / f"sheet_{k // per_sheet + 1}.jpg"
        pair_sheet(cfg, pairs.iloc[k:k + per_sheet],
                   f"{n} random TEST images (seed {cfg['seed']}), each with its most similar VAL image by g "
                   f"- pairs {k + 1}-{min(k + per_sheet, n)}", p)
        sheets.append(p.name)
    q = np.quantile(near["g"], [0.1, 0.5, 0.9])
    summary = {
        "method": f"{n} random test images (numpy default_rng(seed={cfg['seed']}).permutation), each paired "
                  "with its most similar val image by 64x36 gradient-magnitude correlation g (A1 protocol); "
                  "judged by eye by Yash: same / unclear / different",
        "n": int(n), "val_images": len(val), "test_images": len(test),
        "g_drawn": {k: round(float(getattr(pairs["g"], k)()), 4) for k in ("min", "median", "max")},
        "g_nearest_val_all_test": {"p10": round(float(q[0]), 4), "median": round(float(q[1]), 4),
                                   "p90": round(float(q[2]), 4), "count_ge_0.36": int((near["g"] >= 0.36).sum()),
                                   "count_ge_0.48": int((near["g"] >= 0.48).sum())},
        "a1_thumbnail_agreement": a1_agreement(val + test, np.vstack([Tv, Tt])),
        "sheets": sheets, "verdicts": None,
    }
    write_result("leak_test_val/summary", summary, cfg)
    return summary


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k/n (A1's formula)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(float(max(0.0, centre - half)), 3), round(float(min(1.0, centre + half)), 3)


def record_verdicts(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read Yash's verdicts from pairs.csv, store the estimate in summary.json and provenance.json."""
    out = cfg["paths"]["results_dir"] / "leak_test_val"
    pairs = pd.read_csv(out / "pairs.csv", dtype=str, keep_default_na=False)
    v = pairs["verdict"].str.strip().str.lower()
    bad = sorted(set(v) - set(VERDICTS))
    if bad:
        raise ValueError(f"verdicts must be one of {VERDICTS}; found {bad}")
    n, same, unclear = len(v), int((v == "same").sum()), int((v == "unclear").sum())
    res = {"by": "Yash", "n": n, "same_camera": same, "unclear": unclear, "different": n - same - unclear,
           "leakage_estimate": round(same / n, 3), "leakage_95ci": wilson(same, n),
           "leakage_upper_if_unclear_count": round((same + unclear) / n, 3)}
    summary = json.loads((out / "summary.json").read_text())
    summary["verdicts"] = res
    write_result("leak_test_val/summary", summary, cfg)
    prov_path = cfg["paths"]["results_dir"] / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["test_val_leakage"] = {"method": summary["method"], **res,
                                "compare_a1_test_train": {"estimate": 0.175, "ci95": [0.087, 0.32]}}
    write_result("provenance", prov, cfg)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--record", action="store_true", help="record Yash's verdicts from pairs.csv")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke)
    print(json.dumps(record_verdicts(cfg) if args.record else draw(cfg), indent=2))


if __name__ == "__main__":
    main()
