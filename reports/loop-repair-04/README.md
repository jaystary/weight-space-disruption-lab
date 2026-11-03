# Longer-generation published-patch comparison

Completed and audited. Held-out accuracy improved from **7/24 to 22/24** (+62.5 percentage points). All 16 guard answers stayed correct in both arms. The four-comparison adjusted paired p-value was **0.00054932**. [Full report](REPORT.md) · [Aggregate results](aggregate-report.json) · [507-check audit](closure-check.json)

Both arms used the same 16,384-token arithmetic allowance and the same prompts and decoding. The published internal patch was frozen before the fresh questions were generated. The broader goal of discovering a successful edit in this lab remains open. [Protocol](../../docs/loop-repair-04.md) · [Scientific registration](../../registrations/loop-repair-04.json) · [Reference runner](../../tools/weight_space_lab/loop_batch_repair.py)
