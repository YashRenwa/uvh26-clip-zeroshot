# Project rules

Goal: frozen CLIP (zero-shot) classifies UVH-26 vehicle crops (ground-truth ST boxes, 14 classes) so
per-frame counts can be converted to PCU at a camera with no labels, plus a confidence threshold that
decides which crops go to a human. Assignment: BITS AIML ZG525 EC-1 Assignment 2 (Track C).
Student: Yash Renwa, 2025AG05357. Due 11 Nov 2026. Full plan: EXECUTION_PLAN.md (read it every session).

Sister project (read-only): ../uvh26-pcu-counting (Assignment 1). Reuse its split files, PCU factors,
dataset revision and code patterns; never write to it.

## Hard constraints
- Allowed libraries only: numpy, matplotlib, torch, torchvision, opencv, transformers (CLIP),
  huggingface_hub, pandas, pyyaml, tqdm, Pillow, jupyter. No scikit-learn: metrics, bootstrap and
  calibration are written in NumPy.
- Nothing is trained or fine-tuned. Model: openai/clip-vit-base-patch16, always loaded with the
  revision pinned in configs/base.yaml. No hosted API.
- Must run on CPU or a free Colab T4. Every tunable number lives in configs/base.yaml.
- One seed (configs/base.yaml: seed) for random, numpy and torch.
- The test split is sealed. Never score test crops before prompts/ensemble.yaml is frozen (G1) and
  results/frozen_rule.json exists (G2). Test code verifies splits/test.sha256 and runs only with
  EVALUATE_TEST=True. Prompts and the threshold are never changed after test is opened.
- One controlled change: baseline vs ensemble differ ONLY in the text prompts. Image embeddings are
  computed once and shared by both arms; assert it.
- Never call a softmax score a probability, in code comments, outputs or report text.
- Failure examples are drawn at random with the seed, never picked by hand.
- Do not write image embeddings to disk in the notebook run (privacy mitigation); a gitignored dev
  cache under data/cache is allowed locally.
- Every figure that shows image pixels goes through privacy.redact().
- Every script writes its key numbers to results/*.json so the report can quote them.

## Working style
- Smoke-test every script on CPU with 20 val images (--smoke) before full runs.
- Keep the notebook thin: it calls src/ functions and explains each step in markdown mapped to the
  brief's 8 steps.
- Stop at gates G1-G4 (EXECUTION_PLAN.md section 2) and wait for Yash. He makes framing, analysis
  and conclusion calls; Claude proposes options and writes code. Log decisions in DECISIONS.md.
- Ask before: pushing to GitHub, making the repo public, changing a frozen file, or anything on the
  eLearn portal.
- Notebook markdown and comments read as Yash's project notes: no "phase", "smoke test", "grader"
  or agent wording in the submitted notebook.

## Dataset facts (from Assignment 1, confirmed 2026-10-05)
- Hub repo iisc-aim/UVH-26 (dataset, CC BY 4.0), revision 59f82c57821e8a54dc40bc1f42e83909dbad0b70.
- Images at UVH-26-{Train,Val}/data/NNN/<id>.png (not images/). Exact repo paths are in
  splits/pool.csv (repo_path column). PNGs are ~3.4 MB, 1920x1080.
- ST annotations: UVH-26-Train/UVH-26-ST-Train.json, UVH-26-Val/UVH-26-ST-Val.json. The A1 split
  mixes official train and val images, so both JSONs are needed.
- Image ids differ between MV and ST files: join on file_name. category_id k -> class k-1.
- ST JSONs contain exact duplicate boxes; drop them (A1 did).
- Split: val 452 images (355 pseudo-cameras), test 477 (410). test.txt sha256
  f8181d247d7e034f5810b2bb2a978e2739e5055dbc8b3240877c5caa1f667ff6. Residual leakage ~17.5%.
- ST box counts, val / test: Hatchback 619/607, Sedan 379/356, SUV 347/300, MUV 223/212,
  Bus 212/147, Truck 350/339, Three-wheeler 1006/937, Two-wheeler 2981/2816, LCV 329/372,
  Mini-bus 18/25, Tempo-traveller 21/28, Bicycle 97/95, Van 44/62, Others 4/3 (after dedup;
  51 exact duplicates dropped in each split, raw 6,681 / 6,350 boxes).
- Local raw PNGs for the whole A1 pool are in ../uvh26-pcu-counting/data/raw: set UVH_RAW_ROOT to
  that path to skip downloading.
- transformers 5.x: CLIPModel.get_image_features returns an output object; the projected
  embedding is .pooler_output (same for text features: check the attribute).
- A1 reference (test, fine-tuned YOLOv8, includes detection errors): mAP50-95 0.489, PCU MAE 2.61
  per frame (15.8% of mean demand). Quote only as "what labelled training bought".
