# Notes carried into the failure analysis (Phase 6)

Recorded 2026-11-10 at G1, before test was opened (DECISIONS.md #29). No prompt was changed for these.

- On v2's val IRC confusion (results/prompt_versions/v2_confusion_irc.csv), **truck_bus is the weakest
  IRC category**: 217 of 580 correct, mostly lost to lcv (122) and auto (103).
- **Others still takes about 280 val crops (4 true).** Exact counts on v2 val:
  - 14-class argmax: 282 crops predicted Others, mostly Two-wheeler (103), Three-wheeler (97) and Truck (40).
  - Summed-softmax IRC (the rule from Phase 4 on, DECISIONS.md #28): 226 crops predicted tractor, mostly
    auto (85), two_wheeler (74) and truck_bus (41).
  Tractor has the largest PCU factor (4.0), so these errors push PCU up.
