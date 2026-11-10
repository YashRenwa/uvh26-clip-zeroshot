# UVH-26 zero-shot vehicle classification — Execution Plan for Claude Code

BITS Pilani WILP · AIML ZG525 Computer Vision · EC-1 Assignment 2 of 2 (10 marks)
Student: **Yash Renwa, 2025AG05357** · Due **11 Nov 2026 on eLearn** (confirm the portal's cut-off time; plan to upload by the evening of 11 Nov).
Brief: `CV_Assignment_2.pdf` (in this folder). Sister project: Assignment 1 at `../uvh26-pcu-counting` (read-only for this project).

Read `CLAUDE.md` first, then this whole file, before writing any code.

---

## 0. The question in one paragraph

A traffic engineer wants vehicle counts by class, in Passenger Car Units (PCU), at a junction whose camera has **no labelled data**. In Assignment 1 we fine-tuned YOLOv8 on labelled UVH-26 frames. Here we ask: **how good is a frozen CLIP model, zero-shot, at naming the class of each vehicle — and at what confidence should a crop go to a human instead?** We isolate classification by using ground-truth boxes, so every error is CLIP's. The useful output is a review rule: *CLIP handles X% of vehicles alone at Y% accuracy; a human checks the rest; the PCU error left over is Z.*

Track C (vision-language, zero-shot classification). Tracks A and B are ruled out in the report: A compares architectures, which no junction operator needs answered; B produces images, not counts.

## 1. What the brief rewards, and the guard for each

| Marks | Item | Guard in this plan |
| --- | --- | --- |
| 1.0 | Problem and track | Section 0; decision supported = green-time from PCU at an unlabelled camera |
| 1.5 | Data, licence, contamination | A1's pseudo-camera split reused and checksum-verified; CC BY 4.0; dates checked (Phase 0) |
| 1.5 | Frozen baseline | `openai/clip-vit-base-patch16`, revision hash pinned; CLIP's own processor and zero-shot recipe |
| 2.0 | Controlled intervention | Only the text prompts change; image embeddings computed once and shared by both arms |
| 2.0 | Evaluation and confidence | Prompts and threshold chosen on val only; test opened once; reliability diagram, ECE, coverage curve |
| 1.0 | Failure analysis | Seeded random draw of errors (no cherry-picking); Yash tags causes; pattern checked on all errors |
| 0.5 | Ethics and privacy | Closed vocabulary + no stored embeddings (implemented and tested); redaction of displayed crops |
| 0.5 | Report and reproducibility | Fresh-kernel run, seeds, pinned ids, numbers pulled from `results/*.json` |

Marks are lost for: a test set the model has seen, an uncontrolled comparison, a cherry-picked sample, a confidence score called a probability. **Never write "probability" for a softmax score. Call it a score.**

## 2. Who decides what

The brief says the problem choice, analysis and conclusions must be Yash's. Claude Code writes code, runs it, and **proposes**; Yash decides at the gates below. At a gate: stop, print the evidence, ask, and wait.

| Gate | When | Yash decides |
| --- | --- | --- |
| G1 | End of Phase 3 | Final prompt-ensemble list (frozen after this) |
| G2 | End of Phase 4 | Review-rule target accuracy and which arm is the headline (both on val) |
| G3 | Phase 6 | Cause tag for each sampled failure; the named pattern |
| G4 | Phase 8 | Conclusions, "why this track", ethics judgement in the report |

Every decision goes into `DECISIONS.md` (date · decision · by whom · why · rejected alternative). Yash copies these into his Claude Docs Decision Log.

## 3. Fixed design choices (already decided — do not reopen without asking)

1. **Data:** UVH-26 (`iisc-aim/UVH-26`, dataset revision `59f82c57821e8a54dc40bc1f42e83909dbad0b70`, CC BY 4.0). Images in A1's frozen **val (452 images, 355 pseudo-cameras)** and **test (477 images, 410 pseudo-cameras)** lists. A1's train list is not used (nothing is trained).
2. **Labels:** STAPLE (ST) boxes, exact duplicate boxes dropped — A1's headline label set. Join images on `file_name` (ids differ between MV and ST JSONs). MV is not used.
3. **All boxes, no caps.** Every ST box in val and test becomes one crop (about 6.6k val, 6.3k test). Caps would break per-frame PCU. Rare classes are reported with their support; classes with fewer than 30 test crops (Mini-bus 25, Tempo-traveller 28, Others 3) are flagged "low support" in every table and excluded from the headline macro-F1 (also reported with them included).
4. **14 classes** in A1's order: Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, Tempo-traveller, Bicycle, Van, Others.
5. **Model:** `openai/clip-vit-base-patch16` via `transformers`, `CLIPModel` + `CLIPProcessor`, fp32, eval mode, `torch.no_grad()`. Revision = the commit sha from `huggingface_hub.HfApi().model_info(...).sha`, written into `configs/base.yaml` in Phase 0 and passed as `revision=` on every load.
6. **Crop recipe (identical for both arms):** from the original 1920×1080 PNG; expand the box to a square around its centre with side = 1.1 × max(w, h) (10% context), clip to the frame, pad any out-of-frame part with the CLIP mean pixel. Square first, so CLIP's resize + centre-crop never cuts the vehicle off. Then `CLIPProcessor` unchanged.
7. **Scoring:** image and text embeddings L2-normalised; logits = `model.logit_scale.exp()` × cosine; softmax over 14 classes; prediction = argmax; **score** = max softmax. No temperature fitted, no weights changed.
8. **One change only — the text side.**
   - *Baseline:* one prompt per class, `"a photo of a {name}."`, with `{name}` = the dataset class name lowercased, hyphens kept (`"three-wheeler"`, `"lcv"`→ write as `"LCV"`, `"suv"`→`"SUV"`, `"muv"`→`"MUV"`), and `Others` → `"other vehicle"` for grammar only. This is CLIP's own recipe with the dataset's names. Written in `prompts/baseline.yaml` **before any val crop is scored**.
   - *Intervention (prompt ensemble):* per class, a small set of names/descriptions (e.g. LCV → "light commercial vehicle", "small goods truck"; Three-wheeler → "auto-rickshaw", "three-wheeled auto rickshaw"; Two-wheeler → "motorcycle", "scooter") crossed with 5–8 templates (e.g. "a photo of a {}.", "a CCTV image of a {}.", "a blurry photo of a {}.", "a low-resolution photo of a {}.", "a traffic camera photo of a {} on an Indian road."). Class embedding = mean of the normalised text embeddings, re-normalised (CLIP paper's ensembling). In `prompts/ensemble.yaml`.
   - Image embeddings are computed **once** and reused by both arms, so the image side is provably identical. Same crops, checkpoint, revision, processor, device, dtype, batch size, seed.
9. **Labelled-data reference:** A1's fine-tuned YOLOv8 test numbers (from A1's report / notebook outputs) are quoted only as "what labelled training bought", never as the baseline. They include detection errors; ours do not. Say so wherever they appear. Do not rerun YOLO.
10. **PCU:** IRC:106-1990 factors and the class → IRC category map exactly as A1's `configs/base.yaml` (`pcu:` block) and `src/counts.py::pcu_factors` (interpolated by A1 train ST composition). Port the function, then assert the 14 factors equal A1's.
11. **Library list (handout):** numpy, matplotlib, torch, torchvision, opencv, transformers (CLIP), huggingface_hub, pandas, pyyaml, tqdm, Pillow (transformers dependency), jupyter. **No scikit-learn** — metrics, bootstrap, calibration in NumPy.
12. **No hosted API.** Everything runs locally / on Colab. State this in the report.

## 4. Repo layout

```
uvh26-clip-zeroshot/
  CLAUDE.md                 rules (sealed test, one change, allowed libs)
  EXECUTION_PLAN.md         this file
  DECISIONS.md              dated decision log (Claude proposes, Yash decides)
  README.md                 how to run (local and Colab)
  requirements.txt          pinned; Colab-preinstalled lines marked `colab-skip` (as A1)
  configs/base.yaml         every tunable number and path; env-var overrides as A1
  prompts/baseline.yaml     frozen before Phase 2
  prompts/ensemble.yaml     frozen at G1 (sha256 recorded in results/prompts_frozen.json)
  splits/                   copied from A1: pool.csv groups.csv val.txt test.txt test.sha256 + PROVENANCE.md
  src/
    common.py               config, seeding, write_result, versions (adapt A1's)
    data.py                 download (pinned revision) or reuse local raw; ST labels with dedup
    crops.py                square-context crops, manifest, size / IR / occlusion tags
    zeroshot.py             model load (pinned), image embeddings, text classifiers, predict
    metrics.py              accuracy, balanced acc, P/R/F1, confusion, IRC-level, group bootstrap
    pcu.py                  factors (ported from A1), per-image PCU, MAE, bias
    confidence.py           reliability diagram data, ECE, coverage curve, threshold rule
    failures.py             seeded error draw, renders, tag template, pattern tables
    privacy.py              redact(): Haar face blur + manual boxes; closed-vocab guard
    plotting.py             shared figure style (adapt A1's)
  analysis/failure_tags.csv Yash's tags (G3), committed
  analysis/redact.csv       manual redaction boxes (stem, x, y, w, h, note)
  results/                  *.json and *.csv with every number the report quotes
  notebooks/uvh26_clip_zeroshot.ipynb
  report/                   figures, numbers.json, report.html → report.pdf
```

Keep the notebook thin (as A1): it calls `src/` functions, shows outputs, and has one markdown section per brief step (1–8).

---

## 5. Phases

Each phase ends with: a CPU smoke test (`--smoke`, 20 val images) passing, results written to `results/`, a commit, and a short summary to Yash.

### Phase 0 — Setup and provenance (≈45 min)

1. Create the repo (git init; GitHub `YashRenwa/uvh26-clip-zeroshot`, private until submission — ask Yash before pushing or making it public).
2. `uv venv --python 3.12 .venv`, install requirements, register kernel `uvh26-a2`.
3. Copy from `../uvh26-pcu-counting/splits/`: `pool.csv`, `groups.csv`, `val.txt`, `test.txt`, `test.sha256`. Verify `sha256(test.txt) == f8181d247d7e034f5810b2bb2a978e2739e5055dbc8b3240877c5caa1f667ff6`. Write `splits/PROVENANCE.md`: source repo + commit (`ef77f21`), how the split was built (pseudo-cameras by 64×36 gradient-magnitude correlation g ≥ 0.48, purge g ≥ 0.36, seed 87), measured residual leakage (~17.5%, 95% CI 8.7–32%, 40-image audit), official-split leakage (≥ 30.9%).
4. Pin the CLIP revision: fetch the sha, store in config, print it in the notebook's first cell.
5. **Contamination facts — verify and record in `results/provenance.json` with source URLs:**
   - UVH-26 public release: arXiv 2511.02563 (Nov 2025), IISc AIM announcement. Confirm from the paper when the frames were captured.
   - CLIP ViT-B/16 (OpenAI): trained on WIT-400M, collected before the Jan 2021 release (arXiv 2103.00020). So these frames cannot be in its training data.
   - Honest caveat for the report: the *concepts* (auto-rickshaws, Indian buses, CCTV-style images) were certainly in CLIP's web data, so this tests a new camera domain, not unseen classes.
   - Our own leakage: prompts and threshold are chosen on val cameras; test cameras are disjoint up to the residual ~17.5% from A1, which we quote.
   - Note for the report: A1 used CLIP B/32 embeddings only to *compare* grouping methods; the split itself used gradient thumbnails, so no CLIP signal shaped the test set.
6. Write `DECISIONS.md` with the fixed choices in Section 3 dated 10 Nov 2026 (by Yash where he chose; "proposed by Claude, accepted by Yash" otherwise — ask him once to confirm the attributions).

### Phase 1 — Data and crops (≈1.5 h)

1. `data.py`: for each val/test stem, find `repo_path` in `pool.csv`. If `UVH_RAW_ROOT` points at A1's `data/raw` and the PNG exists, read it there; otherwise `hf_hub_download(repo_id, path, repo_type="dataset", revision=<pinned>)`. Same for the ST JSONs (`UVH-26-Train/UVH-26-ST-Train.json`, `UVH-26-Val/UVH-26-ST-Val.json` — the A1 split mixes official train and val images, so both are needed). Download, never upload.
2. Labels: ST boxes per `file_name`, exact duplicates dropped (assert the drop count matches A1's logic on these images), category_id k → class index k−1.
3. `crops.py`: one crop per box with the Section 3 recipe; keep crops in memory or a gitignored cache, never committed. Manifest `results/crops_{val,test}.csv`: crop_id, stem, split, group, class, box (orig px), long side px, size bin (<32, 32–95, ≥96 px long side), IR flag (frame is near-greyscale: mean |R−G|+|G−B| < 2), frame brightness, max IoU with any other GT box in the frame (occlusion proxy).
4. Write `results/data_summary.json` + a class × split count table (crops), mirroring A1's `split_class_counts.csv`. Check totals match A1's ST counts for val/test after dedup.
5. Figure: a 14-class grid of example crops from **val** (seeded random, one per class), passed through `privacy.redact()`.

### Phase 2 — Frozen baseline on val (≈45 min)

1. `zeroshot.py`: `load_model(cfg)` with the pinned revision; `image_embeddings(crops) -> (N, 512)`; `text_classifier(prompt_yaml) -> (14, 512)`; `predict(img_emb, cls_emb) -> (pred, score, softmax)`.
2. Freeze `prompts/baseline.yaml` (commit it) **before** scoring val.
3. Run on val. Write `results/val_baseline.json`: accuracy, balanced accuracy, macro-F1, per-class P/R/F1/support, IRC-category accuracy, PCU MAE. Print the val confusion matrix.
4. Record time per crop and peak memory (useful context, not a Track A requirement).

### Phase 3 — Intervention designed on val → **Gate G1** (≈1.5 h)

1. Claude drafts `prompts/ensemble.yaml` from class definitions only (UVH-26 paper/card descriptions of each class), before looking at val errors. Score it on val.
2. At most **two** further revisions, each driven by the val confusion matrix. Log every version to `results/prompt_log.csv` (version, sha256 of the YAML, val accuracy, macro-F1, IRC accuracy, PCU MAE, one-line reason). Three versions maximum keeps val tuning honest, and the log shows it.
3. **G1:** show Yash the final YAML and the log. On approval, freeze: `results/prompts_frozen.json` with both YAML sha256s. From here `zeroshot.text_classifier` refuses a YAML whose sha does not match.

### Phase 4 — Confidence and the review rule on val → **Gate G2** (≈1 h)

1. `confidence.py`, for both arms on val:
   - Reliability diagram: 15 equal-width score bins, accuracy vs mean score per bin, crop count per bin shown as bars underneath.
   - ECE (15 equal-width bins); also equal-mass bins as a robustness check.
   - Same at **IRC level**: IRC prediction = category with the highest summed softmax; IRC score = that sum. This is the level PCU depends on.
   - Coverage curve: sort by score; for each coverage c, accuracy of the top-c crops (14-class and IRC).
2. **Review rule:** threshold t = lowest score at which accepted crops reach the target IRC-level accuracy on val (default target 95%). Report on val: coverage, accepted accuracy, and PCU MAE if crops below t are corrected by a human (set to truth).
3. **G2:** Yash confirms the target and which arm is the headline (the one better on val IRC accuracy, decided before test). Write `results/frozen_rule.json` (arm, target, t). Commit.

### Phase 5 — Sealed test, opened once (≈45 min)

1. Guard: `EVALUATE_TEST = False` by default. Test code refuses to run unless `prompts_frozen.json` and `frozen_rule.json` exist and `sha256(test.txt)` matches. Log the run time to `results/test_opened.json`. Run once; if it crashes partway, fix the code and rerun — but never change prompts or t after seeing test numbers.
2. For **both** arms on the same test crops: accuracy, balanced accuracy, macro-F1 (with and without low-support classes), per-class P/R/F1 with support, row-normalised confusion (14 classes and IRC categories), PCU MAE per image, MAE as % of mean true PCU per image, PCU bias.
3. **Paired bootstrap over pseudo-camera groups** (1,000 resamples, seeded): ensemble − baseline for accuracy, macro-F1, IRC accuracy, PCU MAE; 95% percentile intervals. Groups, not crops, because crops in one frame/camera are correlated.
4. Confidence on test with t frozen: reliability diagrams, ECE (both arms), and the review rule's test coverage, accepted IRC accuracy, residual PCU MAE, human workload (% of crops sent to review).
5. All of it in `results/test_metrics.json`; the per-crop predictions in `results/test_predictions.csv` (crop_id, true, pred and score per arm — no embeddings).

### Phase 6 — Failure analysis → **Gate G3** (≈1.5 h incl. Yash's tagging)

1. On the headline arm's test errors: seeded random draw of **30** errors (14-class wrong), plus a seeded draw of **10 confident errors** (score ≥ t), because those pass the review rule unseen.
2. Render each through `privacy.redact()`: the crop, a small frame thumbnail with the box, true / predicted / score / size bin / IR / occlusion IoU. One sheet per 10 errors in `results/failures/`.
3. Write `analysis/failure_tags.csv` (rank, crop_id, cause, note) with cause keys: `lookalike` (fine-grained confusion, e.g. sedan↔hatchback, tempo↔mini-bus), `occlusion`, `small`, `ir_night`, `truncated` (cut off at frame edge), `context` (another vehicle fills the crop), `label` (label looks wrong), `other`. Pre-fill a *suggested* cause from the tags, clearly marked as a suggestion. **Yash assigns the final tag.**
4. Pattern check on **all** test crops (not just 30): error rate by size bin, IR flag, occlusion-IoU bin, and by true class, with counts. This is what turns five pictures into a named pattern.
5. The five failures in the report are the **first five of the random draw**, not picked.

### Phase 7 — Ethics and privacy, implemented (≈45 min)

1. **Risk in this system:** CLIP accepts any text. The same code pointed at city CCTV could search "a woman in a red saree on a scooter" and follow people instead of counting vehicles.
2. **Mitigation implemented:** `zeroshot.py` builds the classifier only from the frozen prompt YAMLs (sha checked); no function takes free text; image embeddings stay in memory and are not written to disk in the notebook run; outputs are per-crop class + score and per-frame counts only. A notebook cell proves it: calling with an unfrozen prompt file raises; `results/` contains no embedding arrays (assert).
3. **Displayed crops:** `privacy.redact()` = OpenCV's bundled Haar frontal-face cascade (`cv2.data.haarcascades`) → Gaussian blur, plus manual boxes from `analysis/redact.csv` for number plates. Claude renders every report figure through it and lists the figures for Yash to eyeball once; add boxes for any readable plate. Costed alternative (not implemented): a licence-plate detector, cost = one more model, its own error rate.
4. Keep the report's ethics text under half a page.

### Phase 8 — Notebook run, report, packaging → **Gate G4** (≈3 h)

1. Notebook sections map to brief steps 1–8. First cell: seed, library versions, CLIP model id + revision, dataset revision, device. Colab path: clone repo at a pinned commit, install `requirements.txt` minus `colab-skip` lines, download data from the Hub (no Drive needed). Local path: `UVH_RAW_ROOT` → A1's `data/raw`.
2. **Reported run:** a fresh-kernel run with `EVALUATE_TEST=True`: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=uvh26-a2`, device forced to CPU for determinism (expected tens of minutes). Commit the executed notebook. Keep it under 10 MB (A1's portal limit): downscale embedded images. Optional cross-check on Colab T4 if Yash has time; CLIP numbers can differ slightly across CPU/MPS/CUDA, so report the CPU run.
3. `report/extract_outputs.py` (adapt A1's): pull every number into `report/numbers.json` from `results/*.json`; report text uses only those.
4. Report (HTML → PDF, ≤ 4 pages, one figure per claim, readable without the notebook), A1's style:

| Page | Content | Figure |
| --- | --- | --- |
| 1 | Problem and decision; why Track C not A/B; data, licence, split, contamination | Example crops per class (redacted) |
| 2 | Baseline (pinned), the one change, results table both arms + bootstrap CIs, labelled-data reference | IRC-level confusion, headline arm |
| 3 | Confidence vs correctness; the review rule on test | Reliability diagram + coverage curve |
| 4 | Failures and the named pattern; ethics (≤ ½ page); AI-assistance declaration | Five random failures (redacted) |

5. Claude drafts the factual sections. Sections marked **[YASH]** — why this track, the conclusions, the named failure pattern, the ethics judgement — are drafted as bullet notes for Yash to write in his own words (G4). Name and ID on page 1. AI declaration in A1's format: what AI was used for (code, drafting, figure layout), and which decisions were Yash's (from `DECISIONS.md`).
6. **Number check:** a script compares every number in the report HTML against `numbers.json`; fix any mismatch.
7. Package: `Yash_Renwa_2025AG05357_EC1_A2.ipynb` + `Yash_Renwa_2025AG05357_EC1_A2_Report.pdf`. Ask Yash before pushing, making the repo public, or touching the portal.

---

## 6. Schedule (IST)

| When | Phases | Gate |
| --- | --- | --- |
| Tue 10 Nov, morning–afternoon | 0, 1, 2 | — |
| Tue 10 Nov, evening | 3, 4 | G1, G2 (Yash, ~30 min total) |
| Tue 10 Nov, night | 5 (test opened once) | — |
| Wed 11 Nov, morning | 6, 7 | G3 (Yash tags 30 + 10 failures, ~45 min) |
| Wed 11 Nov, afternoon | 8 | G4 (Yash writes conclusions, ~1 h) |
| Wed 11 Nov, evening | Upload to eLearn | Yash |

If time runs short, cut in this order: equal-mass ECE check → the 10 confident-error draw → the Colab cross-check. Never cut: the sealed test, the bootstrap, the reliability diagram, the random failure draw.

## 7. Done means

- [ ] `test.txt` checksum verified; test opened once (`results/test_opened.json`)
- [ ] CLIP revision and dataset revision printed in the notebook's first cell
- [ ] Both arms share one set of image embeddings (asserted in the notebook)
- [ ] Both arms' numbers on test, side by side, with group-bootstrap CIs
- [ ] Reliability diagram, ECE, coverage curve, review rule with its test coverage and residual PCU error
- [ ] 30 random failures tagged by Yash; pattern backed by error rates over all test crops
- [ ] Closed-vocabulary guard and no-embeddings assert run in the notebook; displayed crops redacted
- [ ] Notebook runs top to bottom on a fresh kernel, outputs saved, < 10 MB
- [ ] Report ≤ 4 pages, every number traced to `results/`, [YASH] sections written by Yash, AI declaration included
