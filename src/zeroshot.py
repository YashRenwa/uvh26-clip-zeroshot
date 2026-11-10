"""Frozen CLIP zero-shot classifier (openai/clip-vit-base-patch16 at the pinned revision).

Image side: square-context crops (src.crops) -> CLIPProcessor unchanged -> get_image_features ->
L2-normalised (N, 512). Computed once per split and shared by both arms.
Text side: a prompt YAML (templates x names per class) -> mean of the L2-normalised text embeddings
per class, re-normalised (CLIP paper's ensembling; with one prompt this is just that prompt).
Scoring: logit_scale.exp() x cosine, softmax over the 14 classes, prediction = argmax,
score = max softmax. The score is a confidence score, not a probability.

Closed vocabulary: classifiers are built only from prompt YAMLs whose sha256 is registered in
results/prompts_frozen.json (or explicitly marked as a val draft during prompt design). No function
takes free text.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.common import ROOT
from src.crops import iter_crops

# Always the real results/ (also under --smoke), so a smoke run cannot register prompts.
FROZEN_PATH = ROOT / "results" / "prompts_frozen.json"


def load_model(cfg: dict[str, Any]):
    """(model, processor) at the pinned revision, fp32, eval mode, on cfg device."""
    from transformers import CLIPModel, CLIPProcessor

    m = cfg["model"]
    model = CLIPModel.from_pretrained(m["id"], revision=m["revision"], dtype=getattr(torch, m["dtype"]))
    processor = CLIPProcessor.from_pretrained(m["id"], revision=m["revision"])
    return model.to(m["device"]).eval(), processor


def _features(out) -> torch.Tensor:
    """transformers 5.x returns an output object; the projected embedding is pooler_output."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10   # bytes on macOS, KiB on Linux


def image_embeddings(cfg: dict[str, Any], model, processor, manifest: pd.DataFrame,
                     cache: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """(N, 512) L2-normalised image embeddings in manifest order, plus timing.

    cache: optional .npz in the gitignored dev cache (keyed by crop ids, model revision and crop
    context). The notebook run passes None, so no embedding is written to disk there.
    """
    key = {"crop_ids_sha": hashlib.sha256("\n".join(manifest["crop_id"]).encode()).hexdigest(),
           "revision": cfg["model"]["revision"], "context": cfg["crops"]["context"]}
    if cache is not None and cache.exists():
        z = np.load(cache)
        if json.loads(str(z["key"])) == key:
            return z["emb"], json.loads(str(z["timing"])) | {"from_dev_cache": True}
    bs, device = cfg["model"]["batch_size"], cfg["model"]["device"]
    embs, batch = [], []
    t0 = time.perf_counter()

    def flush() -> None:
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            f = _features(model.get_image_features(**inputs)).float()
        embs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        batch.clear()

    from tqdm import tqdm

    for _, crop in tqdm(iter_crops(cfg, manifest), total=len(manifest), desc="image embeddings"):
        batch.append(crop)
        if len(batch) == bs:
            flush()
    if batch:
        flush()
    emb = np.concatenate(embs).astype(np.float32)
    secs = time.perf_counter() - t0
    timing = {"crops": len(emb), "seconds": round(secs, 1), "ms_per_crop": round(1000 * secs / len(emb), 2),
              "includes": "PNG decode + crop + preprocessing + forward pass", "device": device,
              "batch_size": bs, "peak_rss_mb": round(_peak_rss_mb(), 1), "torch_threads": torch.get_num_threads()}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, emb=emb, key=json.dumps(key), timing=json.dumps(timing))
    return emb, timing


def embedding_fingerprint(emb: np.ndarray) -> str:
    """sha256 of the embedding bytes: proves both arms used the same image side."""
    return hashlib.sha256(np.ascontiguousarray(emb).tobytes()).hexdigest()


def read_prompts(path: Path, cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Class -> list of prompt strings (templates x names), in config class order."""
    spec = yaml.safe_load(Path(path).read_text())
    classes = cfg["dataset"]["classes"]
    assert set(spec["names"]) == set(classes), "prompt YAML must name exactly the 14 classes"
    templates = spec["templates"]
    out = {}
    for c in classes:
        own = spec.get("class_templates", {}).get(c, [])
        out[c] = [t.format(n) for n in spec["names"][c] for t in templates + own]
    return out


def check_frozen(path: Path, cfg: dict[str, Any], draft_ok: bool = False) -> str:
    """sha256 of the prompt YAML; raises unless it is registered in results/prompts_frozen.json.

    draft_ok=True allows an unregistered YAML (prompt design on val only, Phase 3).
    """
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    frozen_path = FROZEN_PATH
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    registered = {v["sha256"] for v in frozen.get("prompts", {}).values()}
    if sha not in registered and not draft_ok:
        raise PermissionError(f"{Path(path).name} (sha256 {sha[:12]}) is not a frozen prompt file; "
                              "classifiers are built only from frozen prompt YAMLs")
    return sha


def freeze_prompts(path: Path, cfg: dict[str, Any], arm: str) -> dict[str, Any]:
    """Register a prompt YAML's sha256 in results/prompts_frozen.json (refuses to change an arm)."""
    frozen_path = FROZEN_PATH
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {"prompts": {}}
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    prev = frozen["prompts"].get(arm)
    if prev and prev["sha256"] != sha:
        raise PermissionError(f"arm {arm} is already frozen with a different YAML ({prev['sha256'][:12]})")
    frozen["prompts"][arm] = {"file": str(Path(path).resolve().relative_to(ROOT)),
                              "sha256": sha, "frozen_at": prev["frozen_at"] if prev else time.strftime("%Y-%m-%d %H:%M %Z")}
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen, indent=2))
    return frozen["prompts"][arm]


def text_classifier(cfg: dict[str, Any], model, processor, path: Path,
                    draft_ok: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """(14, 512) class embeddings from a frozen prompt YAML, plus its sha256 and prompt counts."""
    sha = check_frozen(path, cfg, draft_ok)
    prompts = read_prompts(path, cfg)
    rows = []
    for c, texts in prompts.items():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(cfg["model"]["device"])
        with torch.no_grad():
            t = _features(model.get_text_features(**inputs)).float()
        t = torch.nn.functional.normalize(t, dim=-1).mean(0)
        rows.append(torch.nn.functional.normalize(t, dim=-1).cpu().numpy())
    info = {"file": Path(path).name, "sha256": sha, "prompts_per_class": {c: len(v) for c, v in prompts.items()}}
    return np.stack(rows).astype(np.float32), info


def logit_scale(model) -> float:
    """CLIP's learned temperature, exp(logit_scale) (100 for this checkpoint); not refitted."""
    return float(model.logit_scale.exp().item())


def predict(img_emb: np.ndarray, cls_emb: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pred class, score = max softmax, softmax (N, 14)) from normalised embeddings."""
    logits = scale * img_emb.astype(np.float64) @ cls_emb.astype(np.float64).T
    logits -= logits.max(1, keepdims=True)
    sm = np.exp(logits)
    sm /= sm.sum(1, keepdims=True)
    return sm.argmax(1), sm.max(1), sm
