# EDOLAB (Python version)

The Python version of [EDOLAB](https://github.com/EvoMindLab/EDOLAB): an object-oriented platform for evolutionary dynamic optimization. It provides the experiment runner,
result storage, statistical analysis and plotting, two dynamic benchmark generators, and a library of published
dynamic optimization algorithms.

It is also the platform of the CEC 2027 Competition on Dynamic Optimization Problems; all competition experiments
must be run on it. It is distributed under the same license as EDOLAB (GNU GPL v3).

## Contents

| Part | Location |
|---|---|
| Platform core: algorithm and problem base classes, FE accounting, metrics, runner, storage, analysis | `core/` |
| Generalized Moving Peaks Benchmark (registered as `GMPB`) | `instances/problems/gmpb/` |
| DynPORTAL (registered as `DPORTAL`) | `instances/problems/dportal/` |
| Moving Peaks Benchmark (`MPB`), Fixed Peaks (`FPs`), GDBG (`GDBG`) | `instances/problems/{mpb,fps,gdbg}/` |
| Published algorithms | `instances/algorithms/` |
| Problem landscape visualization | `scripts/visualize_problem.py` |

Algorithms: ACFPSO, AMLP_RS_CMA_ES, AMLP_RS_CMSA_ESII, AMPDE, AMPPSO, AMPPSO_BC, AMPPSO_GI, AmQSO, AMSO, APCPSO, CDE, CESO, CPSO, CPSOR, DPCPSO, DSPSO, DynDE, DynPopDE, FTMPSO, HIA_DRI_AN, HmSO, IDSPSO, ImQSO, LRPSO, mCMAES, mDE, mjDE, MPSO, mQSO, psfNBC, PSPSO, RAPSO, RPSO, SPMP, SPSO_AP_AD, TMIPSO.

## Installation

Requires Python 3.10 or later. From the `PythonVersion` folder of the EDOLAB repository:

```bash
python -m pip install -e .            # add .[torch] for torch evaluation devices
python -m cli list                    # registered algorithms, problems and metrics
```

## Quick start

One algorithm, 31 independent runs, on a GMPB instance:

```bash
python -m cli run --algorithm mQSO --problem GMPB --runs 31 --max-parallel-runs 8 \
  --problem-param PeakNumber=10 --problem-param Dimension=5 --problem-param ChangeFrequency=5000 \
  --problem-param ShiftSeverity=1 --problem-param EnvironmentNumber=100 \
  --metric-output E_o --output-dir results/raw_runs/mQSO_F2
```

`--algorithm` may be repeated to run several algorithms on the same instance. DynPORTAL instances use
`--problem DPORTAL` and `ComponentNumber` in place of `PeakNumber`. Each result directory holds `summary.sqlite`
(per-run scalars such as the offline error `E_o`) and `trend.sqlite` (per-FE trends such as `current_error`). All output goes to `PythonVersion/results/` (runs in `raw_runs/`, analyses
in `statistical_analysis/`, figures in `plots/`) from whatever folder the command is started; a relative
`--output-dir` that starts with `results/` is placed there too.

Batch experiments are described in a JSONL task file (one JSON object per line, the first one optionally holding
shared defaults):

```jsonl
{"max_parallel_runs": 8, "default_problem": "GMPB", "default_runs": 31}
{"label": "mqso_default", "algorithm": "mQSO"}
```

```bash
python -m cli batch --task-file my_tasks.jsonl
```

Statistical analysis (Friedman test with the Nemenyi post-hoc test, Wilcoxon signed-rank or rank-sum):

```bash
python -m cli sa list-groups --result-dir results/raw_runs/mQSO_F2
python -m cli sa run --result-dir results/raw_runs/mQSO_F2 --method friedman --sa-metric E_o --format csv
```

`python -m cli run --help`, `batch --help`, `sa --help` and `plot --help` list all options.

## Adding an algorithm

```text
instances/algorithms/my_algorithm/
  __init__.py
  algorithm.py      # the algorithm class
  state.py          # population / swarm state
  metrics.py        # algorithm-specific metrics (may be empty)
```

1. Subclass `core.algorithm.base.BaseAlgorithm` and register it with `@algorithm_registry.register("NAME")`.
2. Implement `parameter_specs`, `initialize` and `iterate`; use `self.rng` for all randomness.
3. Evaluate every solution through `problem.evaluate(...)`, so that the platform counts fitness evaluations and
   computes the metrics; after each call, handle an environment change when `recent_change` is set.
4. Add the registry name, module and class name to the table in `instances/algorithms/__init__.py`.

The algorithms in `instances/algorithms/` are working examples; `mqso` is a short one.

## EDOLAB Relay

Relay is a local web console for experiments: it manages the task files in `tasks/`, runs them on this computer or on
SSH servers from a frozen snapshot of the source, follows their progress, imports results, and draws trend and box
plots and statistical analyses. It needs a conda environment named `EDOLAB` with Python 3.11:

```bash
conda create -n EDOLAB python=3.11
conda activate EDOLAB
python startup_remote.py              # opens http://127.0.0.1:8766 ; --port, --no-open-browser, --data-root
```

`tasks/example_gmpb.jsonl` is a small task to try it with. Relay snapshots the files known to git, so run it from a
git checkout. See `apps/remote/README.md` for SSH servers and Docker.

## CEC 2027 Competition

The competition folder `CEC2027_Competition/` sits next to this folder in the EDOLAB repository and runs on this
Python version: the 24 problem instances (F1–F12 on GMPB, P1–P12 on DynPORTAL) with fixed seeds, a runner that writes
the files of a submission, an integrity check of the test problems, two example algorithms and a template. **The test
problems must not be modified**; see [../CEC2027_Competition/README.md](../CEC2027_Competition/README.md).

```bash
cd ..                                  # the root of the EDOLAB repository
python -m CEC2027_Competition.run --algorithm Example_mQSO --smoke
```
