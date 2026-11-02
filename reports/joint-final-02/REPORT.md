# Joint final evaluation

development, coordinator-executed, not independently qualified; isolation_unverified

Frozen final set: 93 groups across 14 dates.

| Arm | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| 2B/baseline | 18.28% | 24.82% | 0.164 |
| 2B/centroid | 47.31% | 36.08% | 0.351 |
| 2B/ridge | 53.76% | 50.73% | 0.444 |
| 4B/baseline | 43.01% | 46.21% | 0.389 |
| 4B/centroid | 38.71% | 35.31% | 0.329 |
| 4B/ridge | 56.99% | 43.76% | 0.388 |
| 4B/edit | 41.94% | 45.10% | 0.383 |
| 9B/baseline | 55.91% | 53.12% | 0.479 |
| 9B/centroid | 51.61% | 53.77% | 0.480 |
| 9B/ridge | 70.97% | 62.67% | 0.583 |

Registered decisions:
- 4B/edit: development_bar_not_met
- 9B/baseline: development_bar_met
- 9B/centroid: development_bar_met
- 9B/ridge: development_bar_met

Limitations:
- source categories are weak labels
- group holdout within earlier period, not a new temporal holdout
- rare classes and few dates limit precision
- lexical grouping does not establish semantic independence
- exploratory model comparisons, no family-wise multiplicity correction
- supervised readout is not direct weight editing
- timed-out editor usage can be unknown
- unrelated capabilities were not evaluated
- not independent qualification
