# Split provenance

Files `pool.csv`, `groups.csv`, `val.txt`, `test.txt` and `test.sha256` are copied unchanged from
Assignment 1 (`../uvh26-pcu-counting/splits/`, commit `ef77f21`). `train.txt` is not copied, because
nothing is trained here.

| | images | pseudo-cameras |
| --- | --- | --- |
| val | 452 | 355 |
| test | 477 | 410 |

`sha256(test.txt) = f8181d247d7e034f5810b2bb2a978e2739e5055dbc8b3240877c5caa1f667ff6`. This is checked
by `src.provenance` and before any test crop is scored. Val and test share no pseudo-camera and no image.

## How the split was built (Assignment 1)

- Pool: 4,000 UVH-26 images (dataset revision `59f82c57821e8a54dc40bc1f42e83909dbad0b70`). The pool
  mixes the official train and val folders (`official_split` column in `pool.csv`).
- Pseudo-cameras: each image becomes a 64×36 gradient-magnitude thumbnail. Two images are linked
  when the thumbnails' correlation is g ≥ 0.48, and a camera group is one connected component. On 88
  hand-labelled pairs this rule had precision 1.00 (AUC 0.97).
- Split: groups were assigned 60/20/20 to train/val/test (seed 87, the best class balance of 50 seeds).
  Then a purge moved every val/test group that had a direct train partner with g ≥ 0.36 into train,
  repeating until no such pair was left. The final split is 3,071 / 452 / 477 images.

## Measured leakage

- Official UVH-26 split: at least 30.9% of the official val images share a pseudo-camera with train
  (247 of 799 pool images, under the g ≥ 0.48 rule). That is a lower bound.
- A1 split, residual: an audit of 40 random test images, each with its most similar train image, found
  7 clear same-camera pairs, which gives **17.5% (95% CI 8.7–32%)**. If the 2 unclear pairs are counted
  as well, the figure is 22.5%.
- For this project the relevant overlap is val vs test, because prompts and the review threshold are
  tuned on val. No group is shared, but the same audit's error rate applies: some test frames may
  come from a camera that also appears in val.

## CLIP and the split

A1 used CLIP ViT-B/32 embeddings only to compare candidate grouping rules. The frozen split uses the
gradient thumbnails, so no CLIP signal shaped which images are in test.
