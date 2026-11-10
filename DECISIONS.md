# Decision log

Format: date · decision · by whom · why · rejected alternative.
"Claude → Yash" = proposed by Claude, accepted by Yash. Attributions confirmed by Yash on 2026-11-10.

| # | Date | Decision | By | Why | Rejected alternative |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-11-10 | Track C: zero-shot CLIP classification of vehicle crops; the decision supported is green-time from PCU at an unlabelled camera | Yash | Answers what a junction with no labels can do; links to A1's PCU counts | Track A (architecture comparison: no operator needs it answered); Track B (produces images, not counts) |
| 1b | 2026-11-10 | Problem: UVH-26 Indian-traffic vehicle classes | Yash | Continues A1's signal-timing question | Banknote recognition |
| 2 | 2026-11-10 | Data: UVH-26 at revision 59f82c5, A1's val (452 img / 355 cams) and test (477 / 410) lists; A1 train not used | Yash | Reuses a checksum-verified, camera-grouped split with measured leakage | The official split (≥ 30.9% camera leakage) |
| 3 | 2026-11-10 | Labels: ST boxes, exact duplicates dropped, join on file_name, category_id k → class k−1 | Claude → Yash | A1's headline label set; isolates classification from detection | MV labels |
| 4 | 2026-11-10 | All boxes, no per-class caps; classes with < 30 test crops flagged "low support" and dropped from the headline macro-F1 (also reported with them) | Claude → Yash | Caps would break per-frame PCU | Class-balanced sampling |
| 5 | 2026-11-10 | 14 classes in A1's order | Claude → Yash | Same output space as A1's PCU pipeline | Merging rare classes |
| 6 | 2026-11-10 | Model: openai/clip-vit-base-patch16 at revision 57c2164 (fp32, eval, no_grad), loaded with `revision=` every time; no hosted API | Claude → Yash | Standard CLIP zero-shot baseline that runs on CPU or a free T4 | Larger CLIP (L/14); hosted APIs |
| 7 | 2026-11-10 | Crop: square around the box centre, side 1.1 × max(w, h), clipped to the frame, padded with CLIP's mean pixel, then CLIPProcessor unchanged | Claude → Yash | CLIP's centre-crop never cuts the vehicle; 10% context | Tight box crop (centre-crop truncates long vehicles) |
| 8 | 2026-11-10 | Scoring: cosine × logit_scale.exp(), softmax over 14 classes, prediction = argmax, score = max softmax; no temperature fitted | Claude → Yash | CLIP's own recipe; nothing trained | Temperature scaling (a fitted parameter) |
| 9 | 2026-11-10 | One controlled change: text prompts only (baseline "a photo of a {name}." vs a prompt ensemble); image embeddings computed once and shared | Claude → Yash | One change makes the comparison attributable | Changing crops or model as well |
| 10 | 2026-11-10 | Prompt tuning on val: a first draft from class definitions plus at most 2 revisions, all logged | Claude → Yash | Limits val overfitting; the log shows it | Unlimited prompt search |
| 11 | 2026-11-10 | A1's fine-tuned YOLOv8 test numbers quoted only as "what labelled training bought" (they include detection errors) | Claude → Yash | Honest reference, not a baseline | Rerunning YOLO |
| 12 | 2026-11-10 | PCU: IRC:106-1990 factors and A1's class map / train-ST composition, ported and asserted equal to A1 | Claude → Yash | Same PCU as A1, so the numbers compare | New factors |
| 13 | 2026-11-10 | Libraries: the handout list only (no scikit-learn); frozen weights (nothing trained); report ≤ 4 pages; reported run from a fresh kernel | Handout | Brief requirement | — |
| 14 | 2026-11-10 | Seed 42 for random, numpy and torch (A1's value) | Claude → Yash | One seed, same as A1 | — |
| 15 | 2026-11-10 | Reported run on CPU | Claude → Yash | Deterministic; CLIP numbers can differ slightly across CPU/MPS/CUDA | MPS or Colab T4 as the reported run |
| 16 | 2026-11-10 | Review rule target: 95% accuracy at IRC level on val | Yash | IRC category is what PCU depends on | 14-class target |
| 17 | 2026-11-10 | Redaction: Haar frontal-face blur in privacy.redact(), plus Yash's manual check of every report figure (manual boxes for readable plates) | Yash | Cheap, in the allowed libraries; the manual check catches what Haar misses | A licence-plate detector (one more model with its own error rate) |
| 18 | 2026-11-10 | Repo name uvh26-clip-zeroshot, private on GitHub until submission; the brief PDF stays out of git | Yash | — | — |
| 19 | 2026-11-10 | Measure test-vs-val camera leakage with A1's protocol: 40 seeded random test images, nearest val image by 64×36 gradient-magnitude correlation g, judged by Yash | Yash | Prompts and threshold are tuned on val, so test-vs-val overlap is what matters here; A1 only measured test-vs-train | Reusing A1's test-vs-train 17.5% |
