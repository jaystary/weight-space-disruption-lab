# Loop-repair-03: registered weight-only replication

Status: running; no efficacy result yet. Four authored preflight tests passed. Published base and edited tensor hashes were verified; a synthetic restoration check reproduced original logits exactly.

[Live aggregate progress](status.json) records the observation time, completed pairs, phase and generated-token count. It is refreshed every 15 minutes while active, and at material state changes. It contains no raw examples or individual predictions.

Development is complete: baseline 2/16 correct, edit 3/16, with three repairs and two regressions. Every error was an unanswered generation at the token cap. This small net gain is below the target and is not a final result. [Development aggregates](development.json). Held-out evaluation continues with the frozen protocol.

Before inspecting final or guard outcomes, a [two-trial goal family](../../registrations/weight-only-goal-01.json) was registered. Trial03 keeps its original raw-p stage rule. Any overall goal claim additionally requires Bonferroni-adjusted p = min(1, 2 × raw p) <= .05. [Conditional longer-generation trial](../loop-repair-04/README.md) is prepared but has not started.

## Numerical structure of the fixed patch

A separately registered CPU-only audit found that over99.999% of squared update magnitude lies in one direction in each edited layer. The two dominant directions have absolute cosine0.849005: related, but not identical. Each layer changes64 columns. [Aggregate geometry](patch-geometry.json). This is parameter-space evidence only, with no behavioural or causal claim. No new model forwards or dataset reads were used. Two cache-layout preparation failures were retained; both occurred before decomposition.

Question: does a published internal-weight edit improve fresh arithmetic performance over the identical unmodified model?

- Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Exact p50c patch from [Takehi, Teoh, Zhang and Khattab](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math), CC BY 4.0. It changes 130,188 BF16 entries in two internal MLP matrices; no embeddings, output head or final norm edits.
- Fresh synthetic arithmetic: 16 development and 48 final questions, plus 16 exact-string guardrail tasks. All choices frozen before final inputs generated. No previous classification data reused.
- Identical prompts, decoding and per-case sampling seeds. Thinking enabled; temperature0.6, top-p0.95, top-k20, presence penalty1.5. Maximum4096 new tokens (512 for guards), one sample per case and arm.
- Primary bar: at least10 percentage points final accuracy gain, one-sided exact paired p<=0.05, no guard accuracy decline, complete execution without failures. Report paired95% interval, repairs, regressions, per-family scores, completion/truncation and repetition measures.
- No gradient training, fitted classifier, adaptive selection or new editor calls. Original failed results remain. Every scored case runs once.

Registration SHA256: `7da7d9527cd16b6427eadd19ab4ec3fd3d2f2d0ea1f75efede9183c054e943fa`. Executed source: `5d07407962668f9bd479e02876cf3f78c9fbaec0`. Source and private run artifacts are retained; the public snapshot excludes raw questions, sampled answers and operating details.

This is a bounded replication of an existing patch, not independent discovery or replication of the paper's benchmark score. The synthetic dataset, Transformers engine, one sample per case and4096-token cap differ from that study. Small task-specific results cannot establish general model improvement. Coordinator-executed; not independently qualified. A future independent edit loop needs a fresh registration and final cohort, with a declared policy for multiple trials.

## Reproduction harness

[Runner](../../tools/weight_space_lab/loop_repair.py), [data generator and scorer](../../tools/weight_space_lab/loop_contract.py), [registration](../../registrations/loop-repair-03.json), and [four synthetic checks](../../tests/test_loop_contract.py) are published. Package and cache names are generalized from the executed source; model, patch, prompts, decoding and decision rules are preserved. The published registration includes projection metadata, so its file hash differs from the original hash above.

Use a CUDA environment under `/workspace/` with Python 3.12, PyTorch 2.8.0+cu128 and the versions in the root `requirements.txt` and `constraints.txt`. From a clean committed checkout, run:

```sh
PYTHONPATH=. python -m pytest -q tests/test_loop_contract.py
python -m tools.weight_space_lab.loop_repair
```

The runner downloads the pinned public model and patch, verifies tensor hashes, freezes choices and writes a new private synthetic cohort before inference. It refuses to overwrite an existing run. A new invocation is a new independent trial, not an exact replay of the private seed or a retry of the reported cases. A fresh registration and multiple-trial policy are required before using further trials as evidence.

All four checks also passed against this public projection on the remote CPU (2.57 seconds; no model inference). An initial source transfer failed on archive ownership metadata before tests began; the corrected transfer excluded that metadata. The live measurement remained unchanged.
