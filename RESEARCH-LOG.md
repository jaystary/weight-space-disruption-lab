# Research log

Development evidence; coordinator-executed, not independently qualified. Labels are source metadata, not independently adjudicated truth. Entries distinguish measured results, inference and proposed tests. Private conversations, raw examples and operating details are excluded.

## Findings — 2026-11-02 UTC

### F01 — Blind sparse edits did not pass the first trial

- Hypothesis: small internal weight edits improve classification under fixed decoding.
- Evidence: three editor sessions selected identity; twelve evaluated patches, one diagnostic-only call, two timeouts. Final baseline: 37.36% accuracy, 42.26% balanced accuracy on 182 groups across eight later dates. All selected edits tied baseline.
- Conclusion: registered bar not met. This restricts the tested search, not all possible weight edits.
- Source: `reports/cause-01/aggregate-report.json`; registration `cause-pilot-01.json`.

### F02 — Internal changes can affect outputs without helping

- Hypothesis: training class-contrast directions reinforce the intended category tokens.
- Evidence: strong contrast changed 62/80 validation predictions and reduced accuracy to 20.0%. On eight fixed training probes its final-layer residual changed by mean relative L2 0.987; seven predictions changed, four correct answers became wrong, none became correct.
- Conclusion: intervention reaches the output; direction or scale is harmful. Weight change alone is not evidence of improvement.
- Source: `reports/cause-02/layer-summary.json`, `cached-diagnostics.json`, `development-summary.json`.

### F03 — Best edit has a small, unconfirmed gain

- Evidence: error-direction ablation raised validation accuracy from 38.75% to 40.0%, balanced accuracy from 45.25% to 45.68%: one extra correct case. Eight training probes showed mean final-layer relative L2 0.00606, no prediction changes.
- Conclusion: too small to establish useful generalization. Final set remains closed.
- Next: separate repaired cases from newly broken cases; inspect class margins and error mechanisms before increasing edit strength.

### F04 — Frozen representations contain useful class information

| Validation arm | Accuracy | Balanced accuracy |
|---|---:|---:|
| 2B base | 17.50% | 21.79% |
| 2B selected ridge | 50.00% | 46.43% |
| 4B base | 38.75% | 45.25% |
| 4B selected ridge | 66.25% | 61.21% |
| 9B base | 50.00% | 48.84% |
| 9B selected ridge | 72.50% | 66.63% |

- Evidence: same 138 training / 80 validation groups and fixed decoder; ridge setting selected by balanced accuracy then accuracy.
- Inference: decision readout is a plausible bottleneck. This does not locate a defective internal circuit or prove larger models benefit more from weight editing.
- Limits: supervised references add a fitted classifier; they are not successful direct weight edits. Validation was used for selection. Shared final 93 groups remain closed.
- Source: `reports/cause-02/development-summary.json`, `reports/model-size-01/2B.json`, `9B.json`.

### F05 — Editor timeouts are execution failures

- Evidence: six adaptive calls produced one diagnostic-only response and five 300-second timeouts, with no evaluated adaptive patch. Usage for timed-out calls is unknown.
- Conclusion: these calls do not test a weight-edit hypothesis. Preserve failures; do not silently increase the frozen timeout or claim the model found no useful edit.
- Source: `reports/cause-02/adaptive-summary.json`.

### F06 — Next hypothesis: locate the failing computation

- Status: proposed, not measured. Parameter norms and numerical rank cannot by themselves prove a route is suppressed; effective behavior depends on activations and downstream transformations.
- Proposed sequence: choose matched correct/incorrect training examples; trace layer outputs and category margins; identify the earliest stable task-relevant divergence; use bounded reversible activation interventions as causal probes; only then translate a supported mechanism into a sparse weight edit.
- Controls: keep unchanged baseline, identity restoration, changed-entry limit, unrelated-error counts and fixed final protocol. A rescued training example is diagnostic evidence, not a generalization result.
- Any additional measurement needs a written scope and bound before execution. Existing registrations and negative results remain unchanged.

## Lookup and continuation

Read this log first, then the cited aggregate files and registrations. Append findings with evidence, interpretation, uncertainty and next action; never erase failed attempts. Freeze all selected models, classifiers and edits before opening the shared final set once. Current evidence does not establish reliable weight-space improvement or preservation of unrelated capabilities.

### F07 — Native gate/value probe rejects blanket removal in the tested slice

- Hypothesis: positive products from negative gate/value factors contribute harmful interference in previously selected final-MLP channels.
- Registered probe: all 138 training records; union of 190 channels from two existing patches; 278 forwards including two synthetic baseline/sham checks. No weight changes, validation access or final access.
- Evidence: double-negative factors account for 29.52% of selected channel/example pairs and 14.65% of absolute product mass. Removing only these products changed six training predictions: one wrong-to-correct, two correct-to-wrong. Median true-class margin change was zero; p10/p90 were -0.125/+0.125.
- Guards passed: native product identity, unchanged sham logits and original training prediction replay.
- Conclusion: this route influences outputs, but blanket removal did not improve this slice. Sign alone is not a harm label. No claim about other channels, token positions, tasks or generalization.
- Source: `registrations/native-gates-01.json`, `reports/cause-02/native-gates-summary.json`.

### F08 — Literature lead and prompt integration

- Primary source: [Goodfire: Neural Networks Are Already Clean](https://already-clean.goodfire.com/), an informal essay and working hypothesis. Practical lead: inspect native components and split gate/value sign regimes rather than judging their product alone. Broad cleanliness claims remain unverified for this checkpoint. [Discussion](https://news.ycombinator.com/item?id=50105647) is context, not replication.
- Harness now has `prompts/mechanistic-researcher-v1.md`, plus a renderer that requires its hash in the registration. The adaptive caller uses it only when explicitly pinned; historical registrations retain their historical prompt. No extra editor call was made.
- Prompt requires falsifiable competing explanations, matched training examples, exact capability boundaries, reversible causal probes, separate repairs/regressions, evidence references and honest unknowns. It does not promise improvement or invent semantic neuron names.
- Next registered loop can pin this prompt. Existing six-call limit and old failures remain intact.

### F09 — Influence map scope

- `docs/influence-map.svg` links actual final-layer interventions, measured state changes, class margins and validation outcomes. Ten candidate rows are drawn directly from aggregate files; internal traces use a separate eight-example training population.
- Earlier layers remain untested intervention sites. A zero upstream change is expected for a final-layer edit, not proof those layers are irrelevant.
- Next useful measurements: context-specific gate routes, matched success/failure traces and discriminating interventions. No full circuit atlas or universal improvement knob has been established.

### F10 — Joint final evaluation: weight edit failed; larger-model comparisons passed their development bars

- Completed once on 93 previously unused groups across 14 dates. Every model, fitted readout and patch was frozen before final records opened. All ten arms completed; consistency audit found no final execution failures. No scored-case retries.
- 4B base: 43.01% accuracy / 46.21% balanced accuracy. Selected error-ablation-2: 41.94% / 45.10%. It repaired zero final errors, broke one correct answer, and changed two predictions. Registered weight-edit bar was not met.
- 9B base: 55.91% / 53.12%; 9B selected ridge: 70.97% / 62.67%. These are model-size and supervised-readout results, not successful weight edits.
- 9B versus corresponding 4B arms met the registered development bar: baseline +6.90 percentage points balanced accuracy, centroid +18.46, ridge +18.91; accuracy did not decline and both paired group/day lower confidence bounds were positive. These exploratory comparisons are not multiplicity-adjusted independent qualification.
- 4B ridge's validation gain did not hold uniformly: final 56.99% accuracy but 43.76% balanced accuracy, below its base on the primary metric. This is why selection-set results were not promoted as final claims.
- Source: `reports/joint-final-02/aggregate-report.json`, `REPORT.md`, `closure-check.json`; registrations `cause-02.json`, `model-size-01.json`, `joint-final-02.json`.
- Limits: weak labels; lexical group separation rather than demonstrated incident independence; rare classes; earlier-period group holdout rather than a fresh temporal holdout. Existing final data is now spent for future tuning claims. Further efficacy claims require fresh evaluation data and a new registration.
- Editor accounting: cause-02 had five timed-out calls without usage, in addition to two such calls in cause-01. These are execution failures, not negative mechanistic tests. New prompt remains available for a fresh registered loop.
### F11 — Original successful mechanism differs from the first task

- Literature finding: the original small-language-model result repaired repeated reasoning by changing MLP output columns in layers 15 and 19. Its non-looping model follow-up did not establish an improvement. Source: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math.
- Our original fixed next-category-token classification did not expose a long-generation loop. Its narrow final-layer feature API and five timed-out adaptive calls also left important mechanisms untested. This explains a design difference; it does not prove why our edits failed.
- Next hypothesis: reproduce a supported loop-removal intervention before attempting a new causal edit. Keep reproduction and independent discovery distinct.

### F12 — Weight-only replication registered and launched; outcome pending

- Registration: loop-repair-03, SHA256 7da7d9527cd16b6427eadd19ab4ec3fd3d2f2d0ea1f75efede9183c054e943fa; source 5d07407962668f9bd479e02876cf3f78c9fbaec0. Four remote authored checks passed.
- Qwen3.5-0.8B baseline versus exact published p50c patch; 130,188 BF16 entries differ in two internal MLP matrices. All base/edited tensor hashes verified; restoring original weights reproduced original logits exactly on the synthetic check.
- Fresh synthetic arithmetic: development16, final48, exact-string guard16. Same prompts/decoding and per-case sampling seeds. Primary threshold: >=10pp final gain, exact paired one-sided p<=.05, no guard accuracy loss, no execution failures. No gradients or fitted classifier.
- Outcome pending. This reuses the authors' patch and changes benchmark, inference engine, sample count and token cap (4096 versus32768). It is a bounded replication, not independent edit discovery or reproduction of the published score. Small synthetic cohort limits generality.
- Runtime note: missing optimized recurrent kernels trigger reference implementations. Monitor budget without changing the running candidate or retrying scored cases.
