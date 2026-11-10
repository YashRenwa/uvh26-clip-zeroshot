# Notes carried into the failure analysis (Phase 6)

Recorded 2026-11-10 at G1, before test was opened (DECISIONS.md #29). No prompt was changed for these.

- On v2's val IRC confusion (results/prompt_versions/v2_confusion_irc.csv), **truck_bus is the weakest
  IRC category**: 217 of 580 correct, mostly lost to lcv (122) and auto (103).
- **Others still takes about 280 val crops (4 true).** Exact counts on v2 val:
  - 14-class argmax: 282 crops predicted Others, mostly Two-wheeler (103), Three-wheeler (97) and Truck (40).
  - Summed-softmax IRC (the rule from Phase 4 on, DECISIONS.md #28): 226 crops predicted tractor, mostly
    auto (85), two_wheeler (74) and truck_bus (41).
  Tractor has the largest PCU factor (4.0), so these errors push PCU up.

## Found while preparing Phase 6 (2026-11-10, after test was opened; nothing changed)

- **Double-labelled vehicles in STAPLE.** Some vehicles carry two ST boxes with IoU >= 0.99 but
  *different* classes (most often Hatchback/Sedan, Hatchback/SUV, LCV/Truck, MUV/SUV). A1's dedup drops
  only identical boxes of the same class, so both survive: 572 test crops (9.1%) in 192 images, and
  679 val crops (10.2%) in 207 images. At most one crop of each pair can be classified correctly (ensemble error rate on these
  crops: 76%), and the vehicle is counted twice in the true PCU. This is a `label` cause candidate and
  affects A1's labels too. Labels are not changed after test was opened.
