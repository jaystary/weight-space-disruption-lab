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
