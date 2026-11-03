# Direct weight editing: loop-repair replication

The classification experiments changed outputs but did not improve final accuracy. A fitted classifier is a different method and does not satisfy the research objective. This round returns to direct internal weight editing.

The original study's strongest result repaired repetitive reasoning in Qwen3.5-0.8B. Its authors also report a negative result on a model without that dominant failure. Our one-token classifier could not exhibit the same generation-loop mechanism. The old adaptive harness also supplied only final-layer activations and had five timeouts in six calls; those execution failures were not valid mechanistic tests.

First measure the exact published two-layer patch against the unchanged checkpoint. Verify all four base/edited tensor hashes. Use fresh procedurally generated arithmetic with exact integer answers and separate exact-string instruction checks. No source-category ambiguity, classifier fitting, gradient updates or prompt changes. Register every choice before inference. Keep all raw records and sampled text on the remote worker.

Primary success: at least ten percentage points final accuracy gain, one-sided exact paired p <= 0.05, no guardrail accuracy decline, and complete error-free execution. Both arms use identical decoding and per-case seeds. Report repairs, regressions, token limits, repeated n-grams and uncertainty even on failure. A token-cap event is a truncation measure, not proof of a reasoning loop by itself.

This is a bounded replication on new synthetic inputs. It is not reproduction of the paper's benchmark score, independent rediscovery of its patch, or proof of general mathematical improvement. The 4,096-token budget differs from the paper's 32,768; Transformer inference, one sample per case and small cohorts are additional limits. All differences are frozen in the registration rather than silently changed after readings.

After this measurement, inspect only development trajectories for a specific failure mechanism. A subsequent independent editing loop needs a new registration, a pinned researcher prompt, richer layer/token activation access, and a fresh final cohort. Its own edit must stay distinct from the published-patch comparator. Never tune on this round's final or guard outcomes. Repeated independent trials would also require a predeclared multiplicity policy; stopping at the first nominally significant final result is not acceptable.

Run on the authorized remote GPU only: `python -m tools.weight_space_lab.loop_repair`. Run the authored tests there before measurement: `pytest -q tests/test_loop_contract.py`. No local model execution or local raw-data transfer.

Upstream: [experiment and exact patch](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math), Takehi, Teoh, Zhang and Khattab; patch CC BY 4.0. Model: Qwen/Qwen3.5-0.8B, Apache 2.0. Existing source and failed outcomes remain unchanged.
