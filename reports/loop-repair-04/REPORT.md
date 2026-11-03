# Published internal edit raised held-out arithmetic accuracy from 7/24 to 22/24

Unchanged Qwen3.5-0.8B answered 7 of 24 fresh arithmetic questions correctly (29.2%). Applying the published changes to the layer 15 and 19 MLP down projections raised that to 22/24 (91.7%). The gain was **62.5 percentage points**. Sixteen answers changed from wrong to right and one from right to wrong. Both arms answered all 16 exact-copy guard questions correctly.

| Paired comparison | Questions | Unchanged | Published edit | Repairs | Regressions |
|---|---:|---:|---:|---:|---:|
| Development arithmetic | 8 | 2 | 8 | 6 | 0 |
| Fresh held-out arithmetic | 24 | 7 | 22 | 16 | 1 |
| Exact-copy guard | 16 | 16 | 16 | 0 | 0 |

The exact one-sided paired McNemar p-value was **0.00013733**. Bonferroni adjustment gave **0.00027466** for the original two-trial arithmetic family and **0.00054932** for the four planned comparisons. The registered paired bootstrap 95% interval for the gain was **37.5 to 83.3 percentage points**. The result clears the 10-point gain, adjusted significance, and guard accuracy criteria on this comparison.

**What changed in the answers.** The baseline reached the 16,384-token cap on 17/24 final questions. The edited model reached it on 2/24. All final answers that completed were correct in both arms. Mean generated length fell from 12,892 to 6,672 tokens. This is consistent with the edit helping the model finish. Attribution to a particular neuron, or to either layer in isolation, would require a different comparison.

**What was tested.** Both arms received identical prompts, batching and decoding settings. They differed only in 130,188 changed BF16 entries across two internal matrices of the pinned checkpoint. There was no gradient training, classifier or editor call in this replication. The earlier 4,096-token trial improved 9/48 to 16/48 and missed significance (raw p = 0.05923). The two trials used different fresh samples and other registered protocol differences, so their difference cannot isolate the effect of generation length.

**Limits.** The edit came from Takehi and colleagues. This provides fresh evidence for their patch on one arithmetic distribution and one exact-copy guard. Astra produced no separately verified synthetic patch in this lab. The original-edit and fact-writing goals remain open. This study was coordinator-executed and has no independent grader isolation.

All 96 generations completed once, with 653,164 generated tokens, zero recorded inference failures, and 507 offline audit checks passed. [Aggregate results](aggregate-report.json) · [Closure audit](closure-check.json) · [Frozen protocol](../../registrations/loop-repair-04.json) · [Exact upstream patch](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/edit.npz)
