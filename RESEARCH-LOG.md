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

### F13 — Published patch has one dominant change direction per layer

- Registered CPU-only audit of two fixed published tensor edits, with all base/edited hashes reverified. No data, GPU forwards, model calls or running weight changes.
- Layer15:64 changed columns;99.99945% of squared update norm lies in the first singular direction; relative update norm10.636%. Layer19:64 columns;99.99915%; relative norm8.106%. Absolute cosine between their dominant directions is0.849005.
- Interpretation: each update is almost rank one, but the layer-specific directions are not identical. This supports testing layer-specific directions and channel contributions; it does not establish what those directions mean or whether they improve answers.
- Preparation failures retained:01 failed on missing offline snapshot documentation;01b assumed a monolithic tensor filename. Both failed before tensor decomposition or data access.01c used the observed singleton shard, preserving all hash checks.
- Source: registrations/loop-patch-geometry-01c.json; source7650163fbba5a0ea4ca81c227afa529fe135a74c; runs/loop-patch-geometry-01c/aggregate-report.json. Main loop-repair-03 measurement remains unchanged at5d07407962668f9bd479e02876cf3f78c9fbaec0.

### F14 — Short-cap development results show completion tradeoffs

- All 16 development pairs completed: baseline 2/16 correct (12.5%), published patch 3/16 (18.75%). Three repairs and two regressions; net gain 6.25 percentage points. No final or guard outcomes inspected.
- Every development error had no parsed answer and reached the 4,096-token cap. All three repairs changed capped baseline generations into correct completed answers; both regressions changed correct completed baselines into capped edited generations. Repeated 8-gram fraction fell from 0.2911 to 0.2091, but repetition reduction did not guarantee a completed answer.
- The authors' [pinned report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md) reports median correct-answer lengths of 5,310.5 tokens for baseline and 9,171.5 for their edit on their practice tasks, using a 32,768-token cap. These are external measurements, not our results. Our smaller cap can materially limit comparability.
- Current measurement remains frozen. A separate longer-cap trial would need a new registration, fresh final data, a multiple-trial policy and remaining budget. Longer generations could also expose new errors; improvement is unproven.
- Source: reports/loop-repair-03/development.json; frozen source5d07407962668f9bd479e02876cf3f78c9fbaec0. Final evaluation continues.

### F15 — Conditional longer-generation trial prepared before final readings

- Trial04 and goal family weight-only-goal-01 registered at source738b2272d286c0edeefec0c95fb6d8bbce78bc90 before trial03 final/guard outcomes were inspected. Both use the same published patch. Trial03 continues unchanged.
- Overall goal claims use Bonferroni correction across the two planned trials: adjusted p=min(1,2*raw p), threshold .05. Original trial03 stage bounds and failed outcomes remain intact. This is an additional goal-level requirement, not a rewrite of its original decision.
- Conditional trial04: fresh8development/24final/16guard; arithmetic cap16384; batch8 SDPA, paired batch seeds, both native/chat EOS. Guard copying has thinking disabled in both arms. These changes mean cross-trial differences cannot isolate token-cap effects.
- Seven focused checks passed on remote CPU in2.98s. Three new public-projection checks also passed in0.94s. Installed model class advertises SDPA and supports logits_to_keep; installed generation code sets that to1. No GPU trial04 measurement has run, and throughput/memory admission remains unverified.
- Start only after trial03 closure if its corrected goal result is insufficient and original remaining budget admits the full bounded run. No new resources; no third trial under this family. Source and registered bounds are retained before measurement.

### F16 — Owner requests two synthetic tracks with a direct weight-programming researcher

- Separate context retrieval (fact present among distractors) from parametric memory writing (fact supplied for editing, absent from later questions). Synthetic facts give known targets and fresh entity/value combinations. Both tracks are designed but unexecuted.
- The researcher must inspect real tensors and write its own internal patch. Existing arithmetic runs reuse a published patch and remain replication controls, not new editor discoveries. A numerical replication pass alone does not fulfill the additional researcher-editing request.
- Added direct-weight-researcher-v2 instructions, explicitly selected by researcher_prompt_id and pinned SHA256. Original v1 remains the default. Two remote prompt checks passed; six focused public-projection checks passed in1.07s including the queued trial's updated timing guard. No new editor call occurred.
- [Retrieval-head work](https://arxiv.org/abs/2404.15574) supplies a causal-analysis lead, not proof that strengthening any head improves our model. [ROME](https://rome.baulab.info/) motivates separate memory-write tests; its [official target routine](https://github.com/kmeng01/rome/blob/main/rome/compute_v.py) uses gradient optimization, so it cannot be imported unchanged under this no-gradient contract.
- Data generation, diagnostic interfaces, candidate budgets and confirmatory statistical plans remain to be implemented/registered. No extra resource or synthetic final run has started. Runtime/editor identity must be verified before describing an active Astra editing loop.


### F17 — Direct researcher access restored; synthetic harness prepared

- One authenticated CLI request pinned gpt-6-astra succeeded: 12,304 input tokens, 4,699 output tokens; no separate returned backend identity. The generated program failed before numerical analysis because API names were not imported. A separately registered binding correction reached tensor loading, then stopped on missing hashlib. No patch or GPU call occurred. Failed attempts remain; partial output is not a completed mechanistic finding.
- Two synthetic tracks now have generators, development-only traces, bounded weight-edit execution, paired scoring, restoration checks and a joint final freeze. All target-model computation remains pending. Thirteen focused checks are covered by the tested preparation: eleven initial checks passed, then updated restoration/accounting tests passed in their focused runs. GPU execution has not been validated.
- New goal-level multiplicity uses four planned comparisons; earlier two-trial decisions are preserved. This tightens the overall claim rule before any current final score has been inspected. No significance claim or synthetic efficacy result exists yet.


### F18 — Tracing and experiment controls checked before target execution

- Native tiny-transformer tracing captured fact-position and attention-head features without changing model output. Authored 512-record fixtures used 9,939–9,986 input tokens under the 16,384 bound. These are instrumentation checks, not a positive target-model result.
- Nineteen focused checks passed in 10.03 seconds. They cover synthetic construction, strict scoring, paired inputs, restoration failure, model-client accounting, prompt selection, bounded researcher context and diagnostic deltas. Long haystacks remain complete in evaluation; only researcher display uses explicitly marked excerpts.
- Both selected patches freeze before final data and before the fixed four-case development intervention map. The map will compare whole-patch activation effects; it cannot identify individual-head causality. Target GPU execution remains queued behind the prior evaluation and its audit.


### F19 — Held-out published-patch replication shows a gain but misses significance

- Fresh final48: baseline9/48=18.75%, published patch16/48=33.33%; gain14.58 percentage points,11repairs/4regressions. Exact one-sided pairedp=.0592346; paired95%bootstrap gain interval[0,.291667]. Original stage bar not met. Earlier two-trial adjustedp=.118469; current four-comparison goalp=.236938.
- Guards7/16 to8/16,4repairs/3regressions. No aggregate accuracy loss in this limited guard task; no broader preservation claim. All160generation attempts completed,495869outputtokens,11023.55seconds; closure passed without inference retries.
- This remains a published-patch replication. It motivates further direct-edit research but does not complete the own-researcher goal. No individual final case was used for subsequent tuning.
- New synthetic loop started after closure. Context pilots scored100% at64/256/512records and therefore expose a ceiling, not evidence that weight editing is ineffective. The memory-writing track and final decisions remain pending.

### F20 — Eight researcher calls inspected weights; neither synthetic track produced a patch

Context retrieval: 8/8 on all three pilots and 16/16 on development. Memory writing: unsupported baseline 0/16. Six programs completed numerical diagnostics (126 logged tensor reads); two were rejected before execution for unapproved imports. No nonempty patch reached evaluation. Both finals and guards stayed sealed. No causal activation map or held-out effect exists for these tracks. All 527 closure checks passed. The eight pinned-model requests returned usage but no independent backend identity. Future diagnostic: verify a nonempty reversible write end to end before studying efficacy; use fresh facts and a shuffled-assignment control.

### F21 — Most printed diagnostic feedback did not reach the next editor round

Post hoc, development only. Four numerical programs with a later round printed 186,000 characters in total. The harness forwarded only the first 2,000 per program: 8,000 characters, or 4.3%. Each track's rejected program tried to redirect stdout using imports the sandbox forbids. The remaining output was retained in the private audit, but the next round did not receive it through this feedback field. Other evidence, including hypotheses and available features, also reached the editor. This identifies an interface limitation; it does not establish why no patch was produced.

For a future registered run, give the editor a bounded structured findings field separate from tensor-read logs, include the import allowlist and demonstrate a nonempty reversible write on an unscored fixture. Keep the original security policy and all eight outcomes.

### F22 — Bounded findings and a reversible weight write reached the normal execution path

The opt-in editor interface keeps four structured numerical findings separately from ordinary logs and supplies the import allowlist. A fixture emitted 90,000 characters before its observation. The next rendered prompt received the exact observation. A sandboxed fixture program changed one BF16 value in the internal feed-forward projection of an eight-dimensional transformer. The maximum absolute forward-output change was 0.775390625. Restoring the base restored both weights and output exactly. The identical fixture patch hash and result reproduced on the curated reference code. Each version passed 12 focused checks. This was an unscored CPU software check using a harness-authored program, with zero editor-model calls. [Receipt](reports/editor-interface-smoke-01/aggregate.json)

### F23 — Independent results separate output-format repair, synthetic repair and patch replay

A pretrained Qwen output-row experiment improved 15/32 to 23/32 by fixing JSON fences. Its earlier internal MLP-column edit was flat. A separate agent repaired a synthetic hidden threshold from 35/64 to 64/64. An independent MATH-500 rerun scored the published edited Qwen checkpoint at 51.0%. Scope, commits and controls are recorded in the [source survey](docs/independent-weight-edit-results.md).

### F24 — Supplied facts changed the first-token ranking on every memory development question

Across all 16 questions without supporting context, the frozen model gave one distinct answer after normalization. The target word appeared in none of the responses. None reached the generation cap. With the correct fact supplied, the saved final activation ranked the correct first token first among 11 distinct answer tokens on 16/16 questions. Without the fact it ranked first on 0/16. Cyclically reassigning targets averaged 0.8 correct with support, across 16 questions. This comparison used cached activations. The supported condition had no generated-answer score. [Aggregate](reports/memory-signal-audit-01/aggregate.json)

The MLP inputs at the last prompt token looked alike across different entities at every inspected layer (median pairwise cosine 0.980–0.997). At the entity token the medians ranged from 0.632 to 0.912. A future experiment could check whether the same entity produces a stable direction across paraphrases before trying to use that direction to write a fact. Cosine alone cannot tell us where a fact is stored or whether a write will work. No weights changed in this diagnostic.


### F25 — The cached entity traces cannot measure stability across wordings

The unsupported memory trace sampled the final token of the entity name. The supported trace sampled the final token of the statement supplying its color. Comparing those activations would compare different positions and different information. Each entity had only one unsupported development wording.

An opt-in development check now compares three inputs per entity. It uses the original question, a second wording about the same entity, and that second wording about a different entity. The distractor shares the last six characters of the name. Its final entity token must have exactly the same token ID and position as in the second wording. At each selected MLP, the check measures how often the original entity activation is closer to the true match than the distractor. Ties are recorded separately. The editor receives the paired vectors and aggregate counts. A positive contrast would justify testing an edit. Recall still requires generating the correct answer after a verified write.

On the pinned Qwen tokenizer, all 16 authored fixture triplets met the token and position controls. Eighteen focused software checks passed on each of the source and public versions. No target-model forward or editor call was made. Activation stability on the pretrained model remains unmeasured. [Tokenizer receipt](reports/entity-key-interface-smoke-01/aggregate.json) · [Harness tests](tests/test_needle_key_panel.py)
