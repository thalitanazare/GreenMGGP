# Corrected NARX study

Completed 1440 runs with 30 seeds per configuration and example.

The equal-call arm uses 4510 fitness evaluations per run (100 initial + 49 × 90 offspring), including unchanged offspring and failed evaluations. Native-arm fitness reuse and offspring counts are retained as a diagnostic. Both use the corrected, common polynomial evaluator.

Corrections use the actual maximum factor delay for estimation and initial history, avoid circular indexing, support delayed mixed products and input-only structures, and retain divergence as failure. Thus these are new experiments, not a correction applied retrospectively to the old search.

Pareto-terms minimises distinct canonical terms; Pareto-nodes minimises all tree nodes. Node-count duplicate removal retains the smallest representation of a canonical structure before survival. Hypervolume and the final τ-rule always use the arithmetic cost of each scenario, so the ablations are assessed in the same error–arithmetic-cost plane.

## Median results (primary weight, equal calls)

```text
                         HV_id    HV_val  sel_C       sel_val      best_val  evaluations
example config                                                                          
E1      Green         0.895884  0.893335    7.0  9.150761e-16  3.261810e-16       4510.0
        MO-lex        0.828853  0.827350    7.0  2.700813e-16  2.616445e-16       4510.0
        Pareto-nodes  0.830519  0.827910    7.0  1.124013e-15  3.182611e-16       4510.0
        Pareto-terms  0.895884  0.893335    7.0  1.028564e-15  3.308918e-16       4510.0
        SO            0.640000  0.640000    9.0  2.675850e-16  2.675850e-16       4510.0
E2      Green         0.878398  0.872242    7.0  2.833623e-02  2.797430e-02       4510.0
        MO-lex        0.702304  0.698585   10.0  2.826234e-02  2.803742e-02       4510.0
        Pareto-nodes  0.811563  0.806590   11.0  2.807144e-02  2.801130e-02       4510.0
        Pareto-terms  0.878107  0.872163    7.0  2.833623e-02  2.811700e-02       4510.0
        SO            0.508178  0.505212   12.0  2.843905e-02  2.843905e-02       4510.0
E3      Green         0.954568  0.893383   16.0  1.419918e+00  7.861862e-02       4510.0
        MO-lex        0.797698  0.000000   16.0  1.411413e+00  1.343651e+00       4510.0
        Pareto-nodes  0.929212  0.867630   14.0  1.424536e+00  8.133334e-02       4510.0
        Pareto-terms  0.954011  0.893383   16.0  1.188852e+00  7.861862e-02       4510.0
        SO            0.735825  0.000000   18.5  1.393727e+00  1.393727e+00       4510.0
```

Validation failures (excluded from validation-error medians, retained as failures in the raw records):

```text
example  config      
E1       Green           0
         MO-lex          0
         Pareto-nodes    0
         Pareto-terms    0
         SO              0
E2       Green           0
         MO-lex          0
         Pareto-nodes    0
         Pareto-terms    0
         SO              0
E3       Green           0
         MO-lex          1
         Pareto-nodes    0
         Pareto-terms    6
         SO              1
```

## Reference-point sensitivity (equal calls, primary weight; Green versus MO-lex)

```text
example  error_reference  green_median  baseline_median      A12
     E1              0.1      0.072000         0.072000 0.154444
     E1              0.2      0.157335         0.155350 1.000000
     E1              1.0      0.893335         0.827350 1.000000
     E2              0.1      0.051828         0.049579 0.894444
     E2              0.2      0.136242         0.122356 1.000000
     E2              1.0      0.872242         0.698585 1.000000
     E3              0.1      0.020194         0.000000 0.965556
     E3              0.2      0.115605         0.000000 1.000000
     E3              1.0      0.893383         0.000000 1.000000
```

## Interpretation

On E1, selected-model validation errors are at machine precision; differences below $10^{-14}$, including changes of rank-based effects at tight hypervolume references, should not be interpreted as practical accuracy differences.
The corrected selected-model validation errors, particularly on E3, must be assessed independently of the historical results. Low identification error and low arithmetic cost do not establish good free-run generalisation. The best validation error available in a returned set is included as a diagnostic; it is not used to alter the predeclared identification-only selection rule. Treat changes to the headline claims as results to inspect, including any loss of advantage in the corrected experiments or ablations. Holm correction covers all metrics/examples/baselines within an arm and weight, with validation failures excluded from validation-error tests and counted separately. The old manuscript and energy measurements remain historical and must not be combined with these selected models.

## Energy deferred

450 selected-model records prepared for the five original energy groups. New energy, timing and CO₂e claims require a new session; no new power measurements have been made.

No significant difference in validation hypervolume between Green and Pareto-terms was detected on any of the three examples after the declared Holm correction. The arithmetic objective therefore has not demonstrated a consistent advantage over term count on this indicator in these settings.

## New energy session

The equal-call selected models were measured on 9 October 2026: 311 valid blocks, 446 valid reconstructions. Canonical Green/SO median per-round ratios are 0.798, 0.665 and 0.893 on E1–E3; only E1/E2 are significant after Holm. The corrected NumPy all-gene simulator increases Green energy on E1 and reduces it on E3. The fitted timing ratio is 0.906; the search weights remain unchanged. Full results and audit are in `energia/equal_calls/`. The main manuscript now uses corrected equal-call results and this new energy session.
