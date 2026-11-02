# Cause-classification weight-space pilot

development, coordinator-executed, not independently qualified; isolation_unverified

Result: **development_pilot_bar_not_met**. 182 final groups over 8 days; 32 development examples used for editing.

| Arm | Accuracy | Balanced accuracy | Macro-F1 | Valid JSON/category |
|---|---:|---:|---:|---:|
| baseline | 37.4% | 42.3% | 0.379 | 100.0% |
| control | 42.9% | 41.5% | 0.406 | 96.2% |
| edit-1 | 37.4% | 42.3% | 0.379 | 100.0% |
| edit-2 | 37.4% | 42.3% | 0.379 | 100.0% |
| edit-3 | 37.4% | 42.3% | 0.379 | 100.0% |

Development-majority reference: accuracy 25.3%, balanced accuracy 10.0%.

All scores are measured against source categories, not independently adjudicated causal truth.

Registered decision bars: `{"balanced_accuracy_gain_vs_base_pp": 0.0, "balanced_accuracy_gain_vs_control_pp": 0.803594955768866, "accuracy_guard": false, "schema_guard": true, "median_group_lower_pp": 0.0, "median_day_lower_pp": 0.0}`

Limitations:
- editor token totals include reported calls only; timed-out calls can have unknown usage
- source categories are not independently adjudicated truth
- rare class has very small final support
- eight final observation days; no seasonal/network-wide generalization claim
- lexical grouping does not prove semantic incident independence
- reference arms have same maximum budget but use fewer target requests; editor costs are additional
- restricted sparse MLP edits only; result does not evaluate all possible weight-space programming
- median CI-bound decision is the registered pilot rule, not a confidence interval for algorithm-wide performance
