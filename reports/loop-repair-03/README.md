# Loop-repair-03: registered weight-only replication

Status: running; no efficacy result yet. Four authored preflight tests passed. Published base and edited tensor hashes were verified; a synthetic restoration check reproduced original logits exactly.

Question: does a published internal-weight edit improve fresh arithmetic performance over the identical unmodified model?

- Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Exact p50c patch from [Takehi, Teoh, Zhang and Khattab](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math), CC BY 4.0. It changes 130,188 BF16 entries in two internal MLP matrices; no embeddings, output head or final norm edits.
- Fresh synthetic arithmetic: 16 development and 48 final questions, plus 16 exact-string guardrail tasks. All choices frozen before final inputs generated. No previous classification data reused.
- Identical prompts, decoding and per-case sampling seeds. Thinking enabled; temperature0.6, top-p0.95, top-k20, presence penalty1.5. Maximum4096 new tokens (512 for guards), one sample per case and arm.
- Primary bar: at least10 percentage points final accuracy gain, one-sided exact paired p<=0.05, no guard accuracy decline, complete execution without failures. Report paired95% interval, repairs, regressions, per-family scores, completion/truncation and repetition measures.
- No gradient training, fitted classifier, adaptive selection or new editor calls. Original failed results remain. Every scored case runs once.

Registration SHA256: `7da7d9527cd16b6427eadd19ab4ec3fd3d2f2d0ea1f75efede9183c054e943fa`. Executed source: `5d07407962668f9bd479e02876cf3f78c9fbaec0`. Source and private run artifacts are retained; the public snapshot excludes raw questions, sampled answers and operating details.

This is a bounded replication of an existing patch, not independent discovery or replication of the paper's benchmark score. The synthetic dataset, Transformers engine, one sample per case and4096-token cap differ from that study. Small task-specific results cannot establish general model improvement. Coordinator-executed; not independently qualified. A future independent edit loop needs a fresh registration and final cohort, with a declared policy for multiple trials.
