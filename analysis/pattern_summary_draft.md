# Failure pattern: DRAFT for Yash to finalise (G3)

Headline arm (ensemble v2) on test. The tags are Yash's; Claude judged all 40 in the planning chat at
Yash's request and Yash reviewed them (DECISIONS.md #31). Numbers: analysis/failure_tags.csv,
results/failure_patterns*.{json,csv}, results/label_check.json.

## Candidate wording (Yash's, to finalise)

> CLIP names what fills the crop, not the vehicle in the box: occlusion, neighbours, frame-edge cuts and
> small size account for 20 of 30 random errors.

**Tally of the 30 random errors:** occlusion 7, small 6, lookalike 5, context (a neighbour fills the
crop) 4, truncated 3, ir_night 2, label 2, other 1. Occlusion + context + truncated + small = 20 of 30.

**Error rates over all 6,299 test crops (IRC error rate, errors / crops):**

| Tag | Group | IRC error rate | Comparison |
| --- | --- | --- | --- |
| Overlap with another GT box (max IoU) | 0 | 22.1% (373 / 1,686) | rises steadily |
| | (0, 0.1) | 24.9% (515 / 2,067) | |
| | [0.1, 0.3) | 27.3% (444 / 1,627) | |
| | ≥ 0.3 | 31.9% (293 / 919) | 14-class: 57.7% (530 / 919) vs 30.3% at IoU 0 |
| Frame edge (within 3 px) | touches edge | 36.2% (367 / 1,013) | inside frame 23.8% (1,258 / 5,286) |
| Size (long side) | 32–95 px | 32.0% (478 / 1,496) | ≥ 96 px 23.8% (1,142 / 4,796); < 32 px 5 / 7 |

Each tag raises the error rate by 1.3–1.5× at IRC level (2× for heavy overlap at 14-class level), so the
pattern holds across all crops, not just the 30 drawn. Caveats: IR frames are too few to name (57 crops,
45.6%); 2 of the 30 are label errors (STAPLE double boxes), and occlusion and duplicate boxes overlap
(a duplicate box has IoU ≈ 1 with its twin).

## Costliest class for PCU: trucks

- Trucks have a **78.2% IRC error rate** (265 of 339), the highest of any class with ≥ 30 crops. Next is
  LCV at 59.1% (220 / 372).
- Trucks are 5.4% of test crops but carry **23.8% of all per-crop PCU error** (434 of 1,823 PCU), more
  than any other class, and they under-count (net −378 PCU): a truck (3.28 PCU) read as an auto loses 1.3
  PCU, as an LCV 1.7, as a car 2.3.
- The review rule does not catch them: of the 29 trucks it auto-accepts, 27 are IRC-wrong.
- Known before test (val, DECISIONS.md #29): truck_bus was the weakest IRC category on val (217 / 580).

## Confident errors (the 10 that pass the rule unseen)

5 of 10 are label errors where CLIP is right (#1, #6, #7, #9, #10). So the accepted IRC accuracy on
test (95.2%) is **likely an underestimate** (label errors in the accepted draw cut against CLIP, though noise can cut both ways): by-eye, at least half of the sampled "errors" in the accepted set are
label errors. The secondary label check gives 95.9% when crops in duplicate groups are excluded.
