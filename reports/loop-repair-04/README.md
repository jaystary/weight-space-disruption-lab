# Prepared longer-generation trial

Status: prepared, not running. Seven focused checks passed on the remote CPU (2.98 seconds). GPU batch admission and efficacy are untested. Current trial03 continues unchanged.

[Protocol](../../docs/loop-repair-04.md) · [Scientific registration](../../registrations/loop-repair-04.json) · [Two-trial goal family](../../registrations/weight-only-goal-01.json) · [Reference runner](../../tools/weight_space_lab/loop_batch_repair.py)

This trial can run only after trial03 closes without an audited, multiplicity-adjusted goal success and remaining authorized budget admits the full run. No added resources. New arithmetic gets a 16,384-token cap; eight prompts run together. All prompt, decoding and batch choices match between baseline and edited arms. The exact published patch stays fixed.

The overall goal now requires a Bonferroni-adjusted p <= .05 across these two planned trials. This family was registered before trial03 final or guard outcomes were inspected. Trial03's original unadjusted stage decision remains unchanged and will be reported separately.

The public registration omits private account limits and operating deadline. A new reproduction must supply authorized limits, a fresh registration and the verified prior-trial closure artifacts before execution. The runner is a reference projection; it has not yet completed a GPU measurement.
