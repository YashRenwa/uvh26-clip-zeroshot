"""Write notebooks/uvh26_clip_zeroshot.ipynb (cells only, no outputs) from the cell list below.

The notebook stays thin: every cell calls src/ functions. Edit this file, not the .ipynb, then run
    python notebooks/build_notebook.py
and execute the notebook on a fresh kernel (outputs are saved in the .ipynb).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("uvh26_clip_zeroshot.ipynb")
CELLS: list[tuple[str, str]] = []


def md(s: str, cell_id: str | None = None) -> None:
    CELLS.append(("md", s.strip("\n"), cell_id))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n"), None))


# --------------------------------------------------------------------------------------------------
md("""
# Zero-shot CLIP for vehicle counts at an unlabelled camera (UVH-26)

AIML ZG525 Computer Vision, EC-1 Assignment 2 · Track C (vision-language, zero-shot classification)
Yash Renwa, 2025AG05357

A frozen CLIP ViT-B/16 names each ground-truth vehicle box in UVH-26 frames as one of the 14 dataset
classes. Per-frame counts become PCU (IRC:106-1990), and a confidence threshold decides which crops a
person checks. Nothing is trained.

- Runs top to bottom on a fresh kernel, on CPU (local) or a free Colab runtime.
- **`EVALUATE_TEST`** (next cell): `True` scores the sealed test split. The prompts (`prompts/ensemble.yaml`)
  and the review threshold (`results/frozen_rule.json`) were fixed on val before test was first opened
  (2026-11-10); this run re-scores test with exactly those files and changes nothing.
- Images are downloaded from the Hugging Face Hub at a pinned revision. Locally, `UVH_RAW_ROOT` can point
  to the PNGs already downloaded for Assignment 1.
- Image embeddings stay in memory: this run writes none to disk (checked in step 8).
- Sections follow the brief's eight steps. Every number in the report is printed in a cell output and
  collected into `report/numbers.json` at the end.
""")

code("""
# ---- Run settings ----
EVALUATE_TEST = True   # opens the sealed test split with the frozen prompts and threshold
REPO_URL = "https://github.com/YashRenwa/uvh26-clip-zeroshot.git"
COMMIT = "main"        # pinned to a commit hash for the submitted run
A1_RAW = "../uvh26-pcu-counting/data/raw"   # local only, relative to the repo: Assignment 1's PNGs (skips the download)
""")

code("""
import os, subprocess, sys, time
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
RUN_STARTED = time.time()
os.environ.setdefault("TQDM_MININTERVAL", "60")   # sparse progress bars keep the saved notebook small


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


if IN_COLAB:
    repo_dir = Path("/content/uvh26-clip-zeroshot")
    if not repo_dir.exists():
        sh(f"git clone -q {REPO_URL} {repo_dir}")
    os.chdir(repo_dir)
    sh("git fetch -q origin")
    is_branch = subprocess.run(f"git rev-parse -q --verify origin/{COMMIT}", shell=True, capture_output=True).returncode == 0
    sh(f"git checkout -q -f --detach {'origin/' + COMMIT if is_branch else COMMIT}")
    # keep Colab's preinstalled torch/numpy/opencv builds; pin the rest
    sh("grep -v colab-skip requirements.txt > /tmp/req-colab.txt && pip install -q -r /tmp/req-colab.txt")
    os.environ["UVH_RAW_ROOT"] = "/content/raw"     # PNGs are fetched from the Hub into here
else:
    if Path.cwd().name == "notebooks":
        os.chdir("..")
    if Path(A1_RAW).is_dir():
        os.environ.setdefault("UVH_RAW_ROOT", str(Path(A1_RAW).resolve()))
sys.path.insert(0, str(Path.cwd()))
if EVALUATE_TEST:
    os.environ["EVALUATE_TEST"] = "1"
print("working directory:", Path.cwd(), "| raw images:", os.environ.get("UVH_RAW_ROOT", "data/raw (Hub download)"))
""")

code("""
import io, json
import numpy as np
import pandas as pd
import torch
from IPython.display import Image as ShowImage, display
from PIL import Image

from src.common import ROOT, load_config, run_header, set_seed

SMOKE = os.environ.get("UVH_SMOKE") == "1"   # quick local check on 20 val images; outputs go to data/smoke/
cfg = load_config(smoke=SMOKE)
set_seed(cfg["seed"])
torch.set_num_threads(max(1, os.cpu_count() or 1))
header = run_header(cfg)
for k, v in header.items():
    if k != "versions":
        print(f"{k:17s} {v}")
for name, version in header["versions"].items():
    print(f"  {name:16s} {version}")
RES = cfg["paths"]["results_dir"]
pd.set_option("display.width", 160, "display.max_columns", 30)


def show(path, width=900):
    \"\"\"Display a saved figure downscaled (keeps the notebook small).\"\"\"
    im = Image.open(ROOT / path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    display(ShowImage(buf.getvalue()))


def recorded(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())
""")

# --------------------------------------------------------------------------------------------------
md("""
## 1. Problem and track

**Decision supported.** Signal green time at a junction is planned from traffic demand in PCU (passenger
car units), which needs per-frame counts by vehicle class. At a new camera there are no labels. Can a
model nobody trained on these cameras turn boxed vehicles into class counts good enough for PCU, and can
its own score say which crops a person must check?

**Why Track C.** A zero-shot vision-language model is the only option that gives class names at a camera
with zero labelled images: the classes are described in words, not learned. Track A (ViT vs CNN) needs a
labelled training set for both models, which is the thing an unlabelled camera lacks. Track B (generation)
produces images, not counts; synthetic traffic would not tell a planner anything about this junction.

**Scope.** Boxes are the dataset's ground-truth boxes (UVH-26-ST), so this measures classification only.
Assignment 1's fine-tuned YOLOv8 on the same split is quoted once, as what labelled training bought; its
numbers include detection errors, ours do not.
""", cell_id="yash-step1")

# --------------------------------------------------------------------------------------------------
md("""
## 2. Data, licence and contamination

- **Source.** `iisc-aim/UVH-26` on the Hugging Face Hub (CC BY 4.0), pinned revision below. 1920×1080
  frames from Bengaluru traffic cameras, captured February 2025, with 14 vehicle classes.
- **Split.** Assignment 1's pseudo-camera split, reused unchanged and checksum-verified: val 452 images,
  test 477 images, no pseudo-camera group in both. Exact duplicate ST boxes are dropped (same rule as
  Assignment 1).
- **Crops.** Each box becomes a square crop with 10% context, padded with CLIP's mean pixel. Crops are
  generated in memory; only a manifest (box, size, overlap and frame-edge tags) is kept.

The cell below rebuilds both crop manifests from the raw annotations and checks they equal the
committed ones, so every later number refers to the same crops.
""")

code("""
from src.common import read_split, verify_test_split
from src.crops import build_manifest

print("splits/test.txt sha256:", verify_test_split(cfg))
splits = {"val": read_split(cfg, "val"), "test": read_split(cfg, "test")}
if SMOKE:
    splits = {"val": splits["val"][: cfg["smoke_images"]]}
manifests = {}
for split, stems in splits.items():
    m, summary = build_manifest(cfg, split, stems)
    committed = pd.read_csv(ROOT / "results" / f"crops_{split}.csv", dtype={"stem": str, "group": str})
    committed = committed[committed["stem"].isin(stems)]
    assert m.to_csv(index=False) == committed.to_csv(index=False), f"{split} manifest differs from the committed one"
    manifests[split] = committed.reset_index(drop=True)
    print(f"{split}: {summary['images']} images, {summary['raw_boxes']} raw boxes, "
          f"{summary['exact_duplicates_dropped']} exact duplicates dropped -> {len(m)} crops "
          f"({m['group'].nunique()} pseudo-cameras); matches the committed manifest")
pd.read_csv(ROOT / "results" / "split_class_counts.csv", index_col=0).T
""")

code("""
prov = recorded("provenance")
leak = prov["test_val_leakage"]
print("pinned revisions:", json.dumps(prov["hub"], indent=1))
print("split:", prov["split"])
lo, hi = leak["leakage_95ci"]
print(f"test-vs-val same-camera audit: {leak['same_camera']} of {leak['n']} seeded test images have a same-camera "
      f"val image -> about {leak['leakage_estimate']:.0%} (95% CI {lo:.0%}-{hi:.0%})\\n  method: {leak['method']}\\n  {leak['finding']}")
pd.Series(prov["contamination"]).to_frame("contamination notes")
""")

md("""
**What CLIP has probably seen.** CLIP ViT-B/16 was trained on 400M web image-text pairs collected before
January 2021; the UVH-26 frames were captured in February 2025 and published in 2025, so these exact
frames cannot be in its training data. Generic Indian street photos with captions such as "auto-rickshaw"
very likely are, which is what zero-shot relies on. Within the project, the prompt wording and the
threshold were chosen on val only; about 10% of test images share a camera with a val image (by-eye
audit above), and the prompts were written from the dataset's class definitions and revised at most
twice, so the effect of that overlap on the test numbers is small.
""", cell_id="yash-contamination")

md("""
Example crops per class (val), one seeded random crop each, faces blurred. The Bicycle example is a car
carrying a duplicate STAPLE box labelled Bicycle; see the label-noise finding in Step 7.
""")

code("""
show("results/figures/val_examples.png")
""")

# --------------------------------------------------------------------------------------------------
md("""
## 3. The frozen baseline

`openai/clip-vit-base-patch16` at a pinned revision, used the way it was published: CLIP's own image
processor (224×224, its normalisation), L2-normalised image and text embeddings, logits = CLIP's learned
logit scale × cosine similarity, softmax over the 14 class prompts, prediction = argmax. Nothing is
fine-tuned, no temperature is fitted, no weights change. The baseline prompt is the standard
`"a photo of a {class name}."`.

For PCU, classes map to IRC:106-1990 categories. The IRC category is the one with the highest **summed**
softmax score over its classes (fixed on val before test was opened); the 14-class metrics use the plain
argmax. The argmax→category version is printed alongside as a secondary column.
""")

code("""
from src.zeroshot import load_model, logit_scale

model, processor = load_model(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"{cfg['model']['id']} @ {cfg['model']['revision']} | {n_params / 1e6:.1f}M parameters, all frozen "
      f"| logit scale {logit_scale(model):.1f} | device {cfg['model']['device']}")
print("baseline prompts:", Path("prompts/baseline.yaml").read_text())
""")

# --------------------------------------------------------------------------------------------------
md("""
## 4. One controlled intervention: the text prompts

The only change between the two arms is the text side. The **ensemble** arm averages, per class, the
text embeddings of six camera-aware templates × several descriptive names written from the dataset's
class definitions (`prompts/ensemble.yaml`, every name cites its row in the dataset card). The image
embeddings are computed **once** below and shared by both arms; the code asserts this.

Prompt design used val only. The baseline's val confusion was already known when the first version was
written; at most two revisions were allowed and the version with the best val IRC accuracy was frozen
(the log below). Both prompt files are registered by sha256 in `results/prompts_frozen.json`; the
classifier refuses any other file.
""")

code("""
pd.read_csv(ROOT / "results" / "prompt_log.csv")
""")

code("""
from src.run_val import print_summary, run_arm
from src.zeroshot import embedding_fingerprint, image_embeddings

val = manifests["val"]
val_emb, val_timing = image_embeddings(cfg, model, processor, val, cache=None)   # in memory only
val_sha = embedding_fingerprint(val_emb)
if not SMOKE:
    assert val_sha == recorded("val_baseline")["image_embeddings_sha256"], "val image embeddings differ from the recorded run"
val_out, val_sm = {}, {}
for arm in ("baseline", "ensemble"):
    val_out[arm], val_sm[arm] = run_arm(cfg, model, processor, val, val_emb, val_timing, ROOT / "prompts" / f"{arm}.yaml")
    assert val_out[arm]["image_embeddings_sha256"] == val_sha   # one set of image embeddings for both arms
    print_summary(cfg, val_out[arm])
    print()
""")

# --------------------------------------------------------------------------------------------------
md("""
## 5. Evaluation on data never tuned on

The review rule is fixed on val first: the lowest IRC score threshold *t* at which the crops accepted
without review reach 95% IRC accuracy (90% and 98% shown only for context). The cell checks that this
reproduces the frozen threshold exactly, then opens test.

Test is scored once per run with the frozen prompt files and threshold; the guard refuses to run unless
`EVALUATE_TEST` is set, `splits/test.txt` matches its checksum, and both prompt files and the rule match
their registered hashes. Confidence intervals are a paired percentile bootstrap over pseudo-camera groups
(1,000 resamples, seeded).
""")

code("""
from src.confidence import run_confidence

conf = run_confidence(cfg, {arm: {"stems": val["stem"].to_numpy(), "true": val["cls"].to_numpy(), "softmax": val_sm[arm],
                                  "prompt_sha256": val_out[arm]["prompts"]["sha256"], "image_embeddings_sha256": val_sha}
                            for arm in ("baseline", "ensemble")})
frozen = recorded("frozen_rule")
for arm, r in conf["arms"].items():
    rule = r["rule"]
    if not SMOKE:
        assert rule["threshold"] == frozen["rules"][arm]["threshold"], "the val rule no longer reproduces the frozen threshold"
    ctx = ", ".join(f"{k}: coverage {v['coverage']:.1%}" for k, v in r["rule_context_only"].items())
    print(f"{arm}: t = {rule['threshold']:.4f} -> {rule['coverage']:.1%} of val crops accepted, accepted IRC accuracy "
          f"{rule['accepted_irc_accuracy']:.1%} | context only: {ctx}")
print("headline arm (chosen on val):", frozen["headline_arm"])
""")

code("""
from src.evaluate_test import evaluate

del val_emb   # val embeddings are no longer needed
first_test_sha = None if SMOKE else recorded("test_metrics")["image_embeddings_sha256"]
test = evaluate(cfg, note="notebook run (fresh kernel)", model=model, processor=processor)
if not SMOKE:
    assert test["image_embeddings_sha256"] == first_test_sha, "test image embeddings differ from the first opening"
    print("test image embeddings identical to the first test opening")
rows = {}
for arm, a in test["arms"].items():
    ci = test["bootstrap"]["per_arm_ci95"][arm] if "per_arm_ci95" in test["bootstrap"] else {}
    rows[arm] = {"accuracy": a["accuracy"], "balanced accuracy": a["balanced_accuracy"],
                 "macro-F1 (headline classes)": a["macro_f1_headline"], "IRC accuracy": a["irc_accuracy"],
                 "IRC accuracy (argmax, secondary)": a["irc_accuracy_from_class_argmax"],
                 "PCU MAE per frame": a["pcu"]["pcu_mae"], "PCU MAE / mean demand": a["pcu"]["pcu_rel_mae"],
                 "PCU bias": a["pcu"]["pcu_bias"], "ECE (IRC score)": a["irc_confidence"]["ece"]}
print(f"{test['split']}: {test['n_crops']} crops, {test['n_images']} images, {test['n_groups']} pseudo-cameras")
pd.DataFrame(rows).round(3)
""")

code("""
b = test["bootstrap"]
if "ensemble_minus_baseline" in b:
    print("ensemble minus baseline, 95% CI over pseudo-camera groups:")
    for k, d in b["ensemble_minus_baseline"].items():
        print(f"  {k:20s} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  (share of resamples above 0: {d['share_resamples_above_0']:.3f})")
    for arm, cis in b["per_arm_ci95"].items():
        print(f"{arm}: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in cis.items()))
pc = pd.DataFrame(test["arms"][frozen["headline_arm"]]["per_class"]).T[["precision", "recall", "f1", "support"]]
print(f"\\nper class, {frozen['headline_arm']} arm on {test['split']} (low-support classes, excluded from the headline macro-F1:",
      ", ".join(test["low_support_classes"]) + ")")
pc.round(3)
""")

md("""
Reference, not a baseline: Assignment 1's fine-tuned YOLOv8 on the same test split reached PCU MAE 2.61 per
frame (15.8% of mean demand), including its detection errors; it needed the labelled training set this
camera does not have.

IRC-level confusion of the headline arm on test:
""")

code("""
from src.report_figures import confusion_irc

if not SMOKE:
    show(confusion_irc(cfg), width=640)
""")

# --------------------------------------------------------------------------------------------------
md("""
## 6. Confidence against correctness

The score is CLIP's softmax value (14-class) or the summed softmax of the predicted IRC category. It is a
ranking signal, not a probability, so it is checked against correctness directly: reliability diagrams
(15 equal-width bins; ECE also with equal-mass bins), and coverage curves (accuracy of the crops kept as the
threshold rises). Test is drawn thick, val thin.
""")

code("""
for f in test["figures"]:
    show(f)
""")

code("""
rows = {}
for arm, a in test["arms"].items():
    r = a["rule"]
    rows[arm] = {"threshold t (from val)": r["threshold"], "coverage (accepted unseen)": r["coverage"],
                 "accepted IRC accuracy": r["accepted_irc_accuracy"], "accepted 14-class accuracy": r["accepted_class_accuracy"],
                 "share sent to review": r["human_workload"], "PCU MAE, no review": r["pcu_no_review"]["pcu_mae"],
                 "PCU MAE after review": r["pcu_after_review"]["pcu_mae"]}
print(f"review rule on {test['split']} (target 95% IRC accuracy on accepted crops, threshold frozen on val):")
pd.DataFrame(rows).round(4)
""")

# --------------------------------------------------------------------------------------------------
md("""
## 7. Failure analysis

Failures are drawn at random with the project seed from the headline arm's test errors: 30 from all
14-class errors, then 10 from the errors the review rule lets through (IRC score ≥ *t*, IRC category
wrong). None is picked by hand. I tagged the cause of each of the 40 by looking at the crop and its frame
(`analysis/failure_tags.csv`); the code's suggested cause was only a starting hint and is kept in a separate
column. The pattern is then checked on **all** test crops using tags the manifest already holds
(overlap with another box, frame edge, size).
""")

code("""
from src.failures import run_failures

fail, tables, picks = run_failures(cfg)
if not SMOKE:
    tags = pd.read_csv(ROOT / "analysis" / "failure_tags.csv")
    print(tags.groupby("set")["cause"].value_counts().unstack(fill_value=0))
    fills = tags[tags["set"] == "random"]["cause"].isin(["occlusion", "context", "truncated", "small"]).sum()
    print(f"\\nrandom 30: occlusion + neighbour context + frame-edge cut + small size = {fills} of 30")
""")

code("""
if not SMOKE:
    from src.report_figures import first_five_failures
    show(first_five_failures(cfg), width=1000)
    show("results/failures/confident_sheet_1.png", width=1000)
""")

code("""
for name in ("occlusion_iou", "frame_edge", "size_bin", "ir", "true_class"):
    if name in tables:
        t = tables[name]
        print(f"IRC error rate over all {test['split']} crops by {name}:")
        display(t[[c for c in ("crops", "irc_errors", "irc_error_rate", "class_error_rate") if c in t]])
""")

md("""
**Label check (secondary, defined after test was opened).** Some vehicles carry two ST boxes with
different classes at IoU ≥ 0.99. This changes no headline number; it shows how much of the measured error
is label noise. Assignment 1's labels carry the same duplicates.
""")

code("""
from src.label_check import run_label_check

if not SMOKE:
    lc = run_label_check(cfg)
    print({s: {k: v for k, v in d.items() if k != "group_size_distribution"} for s, d in lc["splits"].items()})
    display(pd.DataFrame({arm: {"accuracy, all crops": r["all_crops_headline"]["accuracy"],
                                "accuracy, duplicates excluded": r["a_excluding_duplicate_groups"]["accuracy"],
                                "accuracy, any label counts": r["b_lenient_all_crops"]["accuracy"],
                                "IRC accuracy, all crops": r["all_crops_headline"]["irc_accuracy"],
                                "IRC accuracy, duplicates excluded": r["a_excluding_duplicate_groups"]["irc_accuracy"],
                                "accepted IRC accuracy, all crops": r["all_crops_headline"]["accepted_irc_accuracy"],
                                "accepted IRC accuracy, duplicates excluded": r["a_excluding_duplicate_groups"]["rule"]["accepted_irc_accuracy"]}
                          for arm, r in lc["test"].items()}).round(3))
""")

# --------------------------------------------------------------------------------------------------
md("""
## 8. Ethics and privacy

**Risk.** CLIP matches images against any text. The same pipeline pointed at traffic cameras could be
asked for people ("a woman in a red saree riding a scooter") instead of vehicle classes, turning a
counting tool into a tool for finding individuals. Frames also show faces and number plates.

**Mitigations implemented.**
1. Closed vocabulary: the classifier is built only from prompt files registered by sha256; free text is
   refused (demonstrated below).
2. Image embeddings stay in memory in this run; no embedding array is written to disk (checked below). The
   outputs are per-crop class and score and per-frame counts only.
3. Every figure that shows pixels passes through a face blur (OpenCV Haar cascade) plus manual boxes, and I
   checked each report figure by eye (`results/redaction_checklist.csv`).
""")

code("""
from src.privacy import assert_no_embedding_files, closed_vocabulary_demo

print(closed_vocabulary_demo(cfg, model, processor))
n = assert_no_embedding_files(ROOT / "results", ROOT / "report", ROOT / "analysis", ROOT / "prompts", RES)
print(f"no embedding arrays among {n} files in results/, report/, analysis/, prompts/")
cache = cfg["paths"]["cache_dir"]
new = [p for p in cache.rglob("*") if p.is_file() and p.stat().st_mtime >= RUN_STARTED] if cache.exists() else []
assert not new, f"this run wrote to the cache: {new}"
print("nothing was written to the embedding cache during this run")
""")

code("""
if not SMOKE:
    from src.report_figures import redaction_checklist
    display(redaction_checklist(cfg, ["report/figures/failures_first5.png"]))
""")

# --------------------------------------------------------------------------------------------------
md("""
## Numbers for the report

Every number the report quotes is collected from `results/` into `report/numbers.json`.
""")

code("""
if not SMOKE:
    sys.path.insert(0, str(ROOT / "report"))
    import extract_outputs
    extract_outputs.main()
print(f"run time: {(time.time() - RUN_STARTED) / 60:.1f} min")
""")


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    nb.cells = []
    for i, (kind, s, cell_id) in enumerate(CELLS):
        c = nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s)
        c["id"] = cell_id or f"c{i:02d}-{kind}"    # fixed ids: notebooks/replace_markdown.py edits by id
        nb.cells.append(c)
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
