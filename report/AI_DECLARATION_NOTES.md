# AI-assistance notes (input for the report's AI declaration)

Running list of what AI (Claude Code) did and which calls were Yash's. Decisions themselves are in
DECISIONS.md; this file records who did the work behind them.

| Date | Item | AI's part | Yash's part |
| --- | --- | --- | --- |
| 2026-11-10 | Repo setup, pinned revisions, split copy and checksum, provenance facts (Phase 0) | Wrote code and notes; looked up paper dates | Confirmed decision attributions |
| 2026-11-10 | Crop manifest, redaction, example grid (Phase 1) | Wrote and ran the code | Chose Haar + manual check of report figures |
| 2026-11-10 | Test-vs-val camera leakage audit (40 random pairs) | Drew and rendered the pairs; judged all 40 pairs by eye (in the planning chat, at Yash's request) | Asked for the audit; spot-checked the 4 same-camera pairs; decided that the by-eye estimate (≈10%, CI 4–23%) is the one reported and that the g-based count is not an upper bound |
| 2026-11-10 | Frozen baseline on val (Phase 2) | Wrote the zero-shot, metrics and PCU code (PCU ported from A1 and asserted equal); wrote the baseline prompt file from the plan's rule and froze it before scoring; ran val | Set the baseline rule in the plan (CLIP's recipe with dataset names) |
| 2026-11-10 | Prompt ensemble design on val (Phase 3) | Drafted v1 from the card's class table; diagnosed v1/v2 errors per name on val; wrote v2 and v3; applied the selection rule | Set the constraints (14 classes, concrete Others, CLIP-known names, camera templates, ≤ 3 versions), the disclosure and the selection rule; approves the final list at G1 |
| 2026-11-10 | Confidence and review rule on val (Phase 4) | Wrote and ran the reliability, ECE, coverage and threshold code; recomputed both arms under the summed-softmax IRC rule | Approved v2 at G1; set the IRC/PCU rule (#28); sets the target and headline arm at G2 |
| 2026-11-10 | Sealed test run (Phase 5) | Wrote the guarded test code and paired group bootstrap; opened test once (22:05 IST) with the frozen prompts and both frozen thresholds | Froze the rule and headline arm at G2 before test; decided that both arms' rules are reported |
| 2026-11-10 | Failure analysis prep (Phase 6) | Wrote the seeded draws (30 random errors, 10 errors passing the rule), redacted sheets, tag template with suggested causes from tags, pattern tables over all test crops; noticed the double-labelled ST boxes | Assigns every final cause tag and names the pattern (G3) |
| 2026-11-10 | G3 cause tags | Judged all 40 failures by eye (in the planning chat, at Yash's request); validated keys, tallied, reconciled with all-crop tables; fixed the frame-edge flag; wrote the duplicate-label check; drafted the pattern summary | Reviewed and owns the tags; asked for the edge check and the duplicate check; writes the final pattern wording |
