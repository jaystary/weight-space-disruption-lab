# Observed gain; confirmatory bar not met

The exact published internal-weight patch raised held-out accuracy from **9/48 (18.75%) to 16/48 (33.33%)**, a **14.58 percentage-point gain**. It repaired 11 answers and broke 4 previously correct answers.

The exact one-sided paired p-value was **0.05923**, above the original .05 bound. The paired 95% bootstrap interval for the gain was **0 to 29.17 percentage points**. The original two-trial adjustment gives .11847; the broader four-comparison goal family gives .23694. No significance claim or completed-goal claim follows.

Guard accuracy was 7/16 to 8/16, with four repairs and three regressions. This preserves aggregate guard accuracy on this small test; it does not establish unchanged behavior on every example or broader capabilities. Guard accuracy itself was low under the registered thinking/token limits.

All 160 generation attempts completed once, using 495,869 output tokens. Closure auditing passed. Original scores, failures, patch identity, prompts, decoding and sample count were not changed. No scored retries.

This is a replication using the authors' published patch, not a new researcher-discovered edit. The 4,096-token arithmetic cap differs from the authors' longer-generation setup. The result is an encouraging observed signal, but insufficient for the predeclared success rule. The separate synthetic researcher-editing loop is now active.

[Aggregate evidence](aggregate-report.json) · [Registered protocol](README.md)
