# AMLP-RS-CMSA-ESII

Independent registration: `AMLP_RS_CMSA_ESII`. The existing
`AMLP_RS_CMA_ES` directory is unchanged. This implementation follows Ali
Ahrari's distributed MATLAB version 1.0, including the overrides in `main.m`,
not the different GMPB CMA report. Source: `paper/amlp_sources_20261009/amlp_rs_cmsa_esii_matlab/`.
The original notice and required citations are in `LICENSE`.

## Source defaults

| Quantity | Value / source |
|---|---|
| Initial and final population coefficient | 9; lambda = floor(9 sqrt(D)); main.m |
| Parent fraction | 0.5; mu = max(1, floor(0.5 + lambda/2)) |
| Recombination | normalized log(1+mu)-log(j), j=1..mu |
| Elitism | 0 (main.m); only the current generation participates |
| Initial sigma | min(0.1, R0), initially R0=sqrt(D)/2 |
| Initial covariance | diag(domain widths squared); sigma is dimensionless |
| Individual sigma learning | tauSigma = sqrt(0.25/D) |
| Covariance time constant | 1 + D(D+1)/(2 mu_eff) |
| Sigma bias correction | divide weighted geometric mean by geomean(all accepted individual sigmas)/old sigma |
| Prediction | AMLP, max level 10, multiplier 0.3, bounds [1e-7, 0.1] |
| Archive quality tolerance | 0.001 |
| tolHistFun | 1e-5; history length floor(10+30D/lambda) |
| Merge window | one full tolHist window; 10 midpoint probes |
| Archive hill–valley | 10 random probes per neighbour, closest 5 neighbours |
| Condition / spatial tolerance | 1e14 / 1e-12 |
| Stagnation window | floor(120 + 0.2 generation + 30D/lambda) |
| Taboo adaptation | initial distance 1; 10th linear percentile; rate 1/sqrt(D); target global/new fractions 0.5 |
| Rejection reduction | 0.99^(1/D); initialization reduces after 101 rejected centres |
| Change detection in MATLAB main | unknown changes, randomized check probability 1/30; adapted below |

The class exposes the eight principal settings in `parameter_specs()`;
remaining constants retain source values. No configuration enables elitism or
switches to CMA. Equal initial/final population coefficients make the source's
budget-dependent interpolation constant, so no interpolation state is needed.

## Behaviour mapping

| Author source | Port |
|---|---|
| main.m, OptimOption.m, OptimProcess.m | algorithm.py initialization and environment/restart lifecycle |
| SubpopulationCMSA.m, SpecStrParCMSA.m, MutationProfile.m | components/cmsa.py and algorithm.py sampling |
| Prediction.m, DynamicManager.m | components/prediction.py and environment-change hook |
| Subpopulation.m, Restart.m | components/niching.py, sampling, stop/merge/local checks |
| Archive.m, Archiving.m | algorithm.py `_update_archive` |
| UtilityMethods.m | NumPy linear percentile / range |

CMSA uses individual log-normal step sizes, the accepted-sample bias correction,
and covariance averaging of boundary-repaired steps. It does not use CSA,
evolution paths, active negative covariance weights, or a pycma kernel.

There is **one filtered archive**, shared by repulsion and prediction. Archive
quality is compared against completed restart bests; quality pruning precedes
hill–valley classification. Archive insertion requires more than one generation.
Replacement requires improvement greater than tolHistFun; repeated basin hits
increase that basin's taboo distance and decrease the others'.

Preserved source details:

- The subpopulation and restart best are overwritten by the current generation's
  best. They are not best-so-far records. A successful merge assigns the archived
  point even if the current generation is better. Hill–valley probes never become
  the subpopulation best or update its mutation distribution.
- At a detected change, driver.m passes the existing archive to the dynamic
  manager **before** any archive update for the interrupted restart. There is no
  separate all-restart archive and no insertion of its paid prefix.
- Prediction sorts archive entries by fitness, normalizes by domain widths,
  tracks nearest neighbours backwards through consecutive nonempty archives,
  and uses the source startup promotion rule. An empty archive interrupts the
  usable history; no completed restart can mean no prediction.
- Every restart draws a random acceptable centre before applying an available
  AMLP recommendation. The recommended centre is clipped to the domain. Initial
  radius history still derives from the random centre's rejection process.
- maxIterPar is declared in StopCriteria.m but never enforced by the source;
  no generation cap is added. The merge rule requires exactly one candidate
  throughout its window, but does not require its identity to remain constant.
- If no archive member is better, the source clears critical indices but retains
  its previous criticality array. The port preserves that historical maximum
  for the local-convergence test; it does not repel from stale indices.

## Explicit EDOLAB adaptations

1. **Maximization:** reverse all fitness comparisons from the MATLAB minimization
   wrapper. CMSA ranking is descending; source strict/non-strict inequalities are
   preserved. EDOLAB computes E_o and E_bbc from every paid evaluation independently
   of the algorithm's current-generation best bookkeeping.
2. **Known changes:** EDOLAB notifies the algorithm at the exact FE boundary.
   Remove randomized change-detection queries and their RNG draws. The original
   main uses unknown changes on DMMOP; this port evaluates the same search method
   under the notified-change protocol of the platform. It is not a reproduction of delayed detection.
3. **Bounded evaluation:** sample a full generation, evaluate its allowed prefix
   through `problem.evaluate`, and return immediately on change/termination.
   Incomplete generations never update CMSA. Paid prefix values still enter the
   framework metrics, but never the next prediction archive. A complete generation
   ending exactly at the boundary is also discarded as its restart is interrupted.
4. **Interrupted probes:** stop a merge/archive test on the first boundary FE;
   do not commit an unfinished same-basin classification or replacement. Retain
   quality pruning already completed before the probe. The original probe loops
   do not check boundaries internally and can cross them; continuing those loops
   would violate EDOLAB's evaluation lifecycle.
5. **Passive records:** omit the DMMOP dummy archive, peak-ratio output, wall-clock
   timestamps and detection counters, which do not drive search. Keep bounded
   prediction history (maxPL+1 archives) and per-run audit counts. The final budget
   stop does not generate an unused prediction or start another restart.
6. **Numerics:** use NumPy/SciPy and the algorithm's Generator. Weighted sigma
   products are evaluated in logarithms; symmetric eigendecomposition uses eigh.
   MATLAB and NumPy RNG/eigenvector conventions differ, so seed integers do not
   imply identical cross-language trajectories. No eigenvalue clipping or hidden
   recovery strategy is added.

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

## Validation scope

The port was tested by comparing CMSA updates over multiple generations against
a separately written loop/product reference derived from MATLAB, and AMLP against
binomial finite differences. The tests also covered sampling order, taboo
rejection, boundary relocation, archives, interruptions, stopping, best records,
and deterministic runs on both benchmarks, and distinguished the new algorithm
from AMLP_RS_CMA_ES.

No MATLAB or Octave executable was found on the development host. These are
**source-derived comparisons**, not executions of the original MATLAB program.
Published numerical-table matching is not a validation requirement of this port.
