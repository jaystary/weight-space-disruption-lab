# Synthetic retrieval and fact writing: no weight edit reached evaluation

The editor inspected real checkpoint tensors. It produced **no saved patch in eight requests**. Neither held-out set nor any guard set opened. There is no edited-model accuracy, paired p-value or intervention map for either track.

| Track | Unchanged development accuracy | Logged tensor reads | Programs that only diagnosed | Programs rejected before execution | Patches evaluated |
|---|---:|---:|---:|---:|---:|
| Retrieve a fact present in the prompt | 16/16 | 51 | 3 | 1 | 0 |
| Recall an inserted fact without support in the prompt | 0/16 | 75 | 3 | 1 | 0 |

The retrieval pilots scored 8/8 at each of 64, 256 and 512 records. Chosen development difficulty had no accuracy headroom. The memory baseline left room for improvement, but no fact was written: its baseline score does not measure edited memory. Each track's second program requested an import outside the sandbox allowlist. The sandbox refused it. Six other programs completed numerical diagnostics. Both selections froze at the unchanged baseline.

The researcher requests used the pinned GPT-6 Astra model identifier. All eight returned token usage; the responses did not include independent backend identity. The target ran on GPU. All 527 offline closure checks passed; 108 model requests, including 56 generations, produced 226 generated tokens. An audit pass verifies the recorded procedure, not efficacy.

**Harness finding (post hoc).** Four diagnostic programs had a later editor round. They printed 186,000 characters combined; the feedback field passed along only the first 2,000 from each (8,000 total, 4.3%). Both rejected programs tried to redirect printed output with disallowed imports. The full output stayed in the private record. This identifies a feedback bottleneck; it does not show what the missing output would have changed.

Before another scored trial, test one nonempty reversible write on an unscored fixture. Return bounded structured findings separately from tensor-read logs so the next round sees them. Then test fresh fact recall against a shuffled-assignment control. These are proposals, not results.

[Numeric evidence](aggregate-report.json) · [Registered protocol](../../registrations/synthetic-needles-01.json) · [Research design](../../docs/synthetic-needle-research.md)
