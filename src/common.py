"""Shared plumbing: config loading, path resolution, seeding, results output, versions."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"

# Env vars that override config paths.
PATH_ENV = {
    "raw_root": "UVH_RAW_ROOT",
    "cache_dir": "UVH_CACHE_DIR",
    "results_dir": "UVH_RESULTS_DIR",
}


def load_config(path: str | Path = DEFAULT_CONFIG, smoke: bool = False) -> dict[str, Any]:
    """Load the YAML config and resolve every entry in `paths` to an absolute Path.

    smoke=True sends results and cache to data/smoke/ so short CPU checks never touch real outputs.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg["paths"].items():
        value = os.environ.get(PATH_ENV.get(key, ""), value)
        p = Path(value).expanduser()
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    cfg["model"]["device"] = os.environ.get("UVH_DEVICE", cfg["model"]["device"])
    if smoke:
        smoke_dir = ROOT / "data" / "smoke"
        cfg["paths"].update(results_dir=smoke_dir / "results", cache_dir=smoke_dir / "cache")
    cfg["smoke"] = smoke
    return cfg


def add_common_args(ap) -> None:
    """CLI flags shared by every src/ script."""
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--smoke", action="store_true", help="20 val images, outputs under data/smoke/")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) with the one project seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_result(name: str, data: dict[str, Any], cfg: dict[str, Any]) -> Path:
    """Write `data` to results/<name>.json so the report can quote it."""
    out = cfg["paths"]["results_dir"] / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return out


def file_sha256(path: str | Path) -> str:
    """sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_split(cfg: dict[str, Any]) -> str:
    """Raise unless splits/test.txt matches the sha256 frozen in the config and splits/test.sha256."""
    path = cfg["paths"]["splits_dir"] / "test.txt"
    got = file_sha256(path)
    recorded = (cfg["paths"]["splits_dir"] / "test.sha256").read_text().split()[0]
    if not (got == recorded == cfg["dataset"]["test_sha256"]):
        raise RuntimeError(f"test.txt sha256 {got} does not match the frozen value")
    return got


def read_split(cfg: dict[str, Any], name: str) -> list[str]:
    """Image stems in splits/<name>.txt (val or test)."""
    return (cfg["paths"]["splits_dir"] / f"{name}.txt").read_text().split()


def library_versions() -> dict[str, str]:
    """Versions of every library the project uses, for the notebook's first cell."""
    import importlib.metadata as md
    import platform

    pkgs = ["numpy", "matplotlib", "torch", "torchvision", "opencv-python", "transformers",
            "huggingface-hub", "pandas", "pyyaml", "tqdm", "pillow"]
    versions = {"python": platform.python_version()}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = "not installed"
    return versions


def run_header(cfg: dict[str, Any]) -> dict[str, Any]:
    """Seed, versions, pinned model and dataset revisions, device: printed in the notebook's first cell."""
    return {
        "seed": cfg["seed"],
        "model": cfg["model"]["id"],
        "model_revision": cfg["model"]["revision"],
        "dataset": cfg["dataset"]["repo_id"],
        "dataset_revision": cfg["dataset"]["revision"],
        "device": cfg["model"]["device"],
        "versions": library_versions(),
    }
