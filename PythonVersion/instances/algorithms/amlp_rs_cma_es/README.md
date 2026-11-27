# AMLP-RS-CMA-ES in EDOLAB

Registry name: `AMLP_RS_CMA_ES`.

This implements the GMPB configuration described in Ahrari et al., EvOpt Report
2022002, using the author's public AMLP-RS-CMSA-ESII MATLAB source for the shared
prediction and repelling-restart mechanisms. It is a source-based port with a
report-based CMA/GMPB reconstruction. The exact original GMPB executable was
not located; this implementation does not claim bitwise parity with it.

Source provenance, URLs, SHA-256 hashes and the search record are in
`docs/AMLP_SOURCE_AUDIT_20261009.md`. The original downloads are preserved under
`paper/amlp_sources_20261009/`. Derived code uses CC BY-NC-SA 4.0; see LICENSE.

## Source-to-code map

| Author's source | EDOLAB implementation |
|---|---|
| Prediction.m; DynamicManager.m | components/prediction.py |
| Restart.m; Subpopulation.m taboo geometry/repair | components/niching.py |
| driver.m; OptimProcess.m; stop/merge/archive routines | algorithm.py |
| CMA core absent from the public package | components/cma.py and shared CMAESOperator |
| Mutable restart and archive records | state.py |

The static search runs one active subpopulation at a time. Completed restarts
contribute archive representatives that repel later samples through distances
defined by the current subpopulation's covariance. This is distinct from a
concurrent population scheduler or pairwise population deletion.

## GMPB report configuration

- Population size: `floor(6*sqrt(D))`, using the shared RS default.
- Parents: `max(1, floor(0.5 + 0.5*lambda))`.
- Positive weights: normalized `log(1+mu)-log(1:mu)`.
- Maximum prediction level: 10.
- Initial normalized sigma: `min(R0, 0.15)`.
- Prediction sigma: `clip(0.5*estimated_error_norm, 0.15e-6, 0.15)`.
  Error is measured after normalizing each coordinate by its domain width.
  There is no extra division by `sqrt(D)`.
- CSA damping: fourth root of the ordinary damping.
- History size: `floor(10+30*D/lambda)`.
- Function-history tolerance: **100**, as printed in the report, not `10^0`.
- Merge window: one complete function-history window.
- Maximum generations: `ceil((70+35*(D+3)^2)/sqrt(lambda))`.
- Other RS stopping, criticality, taboo-distance and boundary rules follow
  the shared source. All change notifications are provided by EDOLAB.

Initial covariance is `diag(domain_widths**2)`, so sigma is dimensionless.
Prediction positions are converted back to decision coordinates and clipped
to the domain. Random offspring use the source's coordinate-wise randomized
boundary relocation, not reflection or pycma's BoundTransform.

## Reconstruction decisions

1. **CMA core.** The public package references, but does not provide,
   `CoreSearchCMA`, `SpecStrParCMA`, or `SubpopulationCMA`. The implementation
   uses EDOLAB's positive-weight, rank-one/rank-mu CMAESOperator with its
   standard path and learning-rate formulas, the author's recombination
   weights, and the report's fourth-root damping. A reference comparison against the missing original core is
   therefore unavailable.
2. **Two archives.** The report explicitly retains every restart's best
   solution for subsequent prediction, while the available static archive
   filters and deduplicates desirable solutions. The implementation preserves
   that static archive and its radius adaptation, and maintains a separate
   all-restart prediction archive. Nearest-neighbour tracking operates on the
   latter, sorted by preceding-environment fitness. The report does not state
   whether its original GMPB implementation used this exact separation.
3. **Interrupted restart.** At a change, the best solution from the paid
   prefix of the active restart is included in prediction history. Incomplete
   generations do not update CMA state. No fitness from the preceding
   environment is used for selection in the new environment.
4. **Best solution.** The reconstructed CMA restart retains its best evaluated
   solution across generations. Source hill-valley probes retain their role
   as niching tests, rather than changing the optimizer's CMA distribution.
5. **Iteration ceiling.** The public CMSA package declares a ceiling parameter
   but never tests it. The explicit whole-fraction ceiling in Report2022002
   is enforced here.

These decisions are frozen before the comparison experiments. Performance
results describe this EDOLAB implementation, not a reproduction of the 2022
published numerical table.

## EDOLAB usage

After the repository's EDOLAB environment startup steps:

```bash
python -m cli run --algorithm AMLP_RS_CMA_ES --problem GMPB \
  --runs 1 --max-parallel-runs 1 --problem-seed 3 --algorithm-seed 101 \
  --problem-param Dimension=5 --problem-param PeakNumber=3 \
  --problem-param ChangeFrequency=401 --problem-param EnvironmentNumber=3 \
  --problem-param EvaluationDevice=numpy --metric-output E_o,E_bbc,T_r
```

All search and hill-valley fitness calls use `problem.evaluate`. Prediction,
archive matching and rejected proposals consume no FE. The algorithm does not
inspect problem optima, future environments or shift severity. Its final
`amlp.audit` event records completed restarts, stop reasons and selected levels.

## Algorithm metrics

`metrics.py` defines four environment trends. The restart lifecycle keeps the
counts, and `prepare_environment_metrics` copies them to `metric_state` at each
environment's last paid FE, before the change is handled. Selecting them does
not change the search.

| Metric | Value at the end of each environment |
|---|---|
| `amlp_archive_size_by_environment` | archived restart bests that repel later samples |
| `amlp_completed_restarts_by_environment` | restarts that met a stop or merge criterion |
| `amlp_predicted_restarts_by_environment` | restarts that began at a predicted position |
| `amlp_mean_prediction_level_by_environment` | mean level of those predictions; NaN if none |
