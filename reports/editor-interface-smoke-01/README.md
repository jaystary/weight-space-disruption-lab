# Editor feedback and reversible-write fixture

One internal BF16 feed-forward weight changed in a small transformer on CPU. The resulting output changed by 0.775390625 at its most affected coordinate. Both original weights and original output restored exactly. A numerical finding reached the next rendered editor prompt after 90,000 characters of diagnostic output. The fixture code authored the patch. There was no scored task or live editor call.

[Fixed checks](../../registrations/editor-interface-smoke-01.json) · [Aggregate receipt](aggregate.json) · [Fixture source](../../tools/weight_space_lab/needle_fixture_smoke.py)

To reproduce, use a fresh isolated Linux worker checkout under `/workspace/` with the sandbox dependencies. Run the following as root inside the isolated worker. The fixture refuses to reuse an existing snapshot or jail directory.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m tools.weight_space_lab.needle_fixture_smoke
```
