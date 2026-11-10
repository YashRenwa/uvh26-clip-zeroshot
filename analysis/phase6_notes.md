# Notes carried into the failure analysis (Phase 6)

Recorded 2026-11-10 at G1, before test was opened (DECISIONS.md #29). No prompt was changed for these.

- On v2's val IRC confusion (results/prompt_versions/v2_confusion_irc.csv), **truck_bus is the weakest
  IRC category**: 217 of 580 correct, mostly lost to lcv (122) and auto (103).
- **Others/tractor still takes 283 val crops (4 true)**, mostly two-wheelers (74) and auto-rickshaws (85).
  Tractor has the largest PCU factor (4.0), so these errors push PCU up.
