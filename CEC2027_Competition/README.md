# CEC 2027 Competition on Dynamic Optimization Problems

This folder is the competition edition of EDOLAB. It sits next to `PythonVersion/` in the EDOLAB repository and runs
on that Python version: it fixes the 24 problem instances, runs an algorithm on them with the official seeds and writes
the files of a submission.

```text
EDOLAB/
├── PythonVersion/          the Python version of EDOLAB (platform, benchmarks, algorithms, Relay)
└── CEC2027_Competition/    this folder
```

## Do not modify the test problems

> **Participants must not modify the protected files.** They are the platform core (`PythonVersion/core/`), the
> benchmark generators GMPB (`PythonVersion/instances/problems/gmpb/`) and DynPORTAL
> (`PythonVersion/instances/problems/dportal/`), the problem definitions (`CEC2027_Competition/instances.py`,
> `problems.py`) and the competition scripts (`run.py`, `verify.py`, `registration.py`, `__init__.py`).

Their SHA-256 digests are listed in `protected.sha256`. The competition runner checks them before every run and
refuses to start when any file differs; the digest of the manifest is written into every result bundle, and the
organizers rerun submitted code on an unmodified copy. To check by hand:

```bash
python -m CEC2027_Competition.verify
```

The problems are black boxes: an algorithm may use the dimension, the search range (`problem.bounds`) and the fitness
values it evaluates, and nothing else of GMPB or DynPORTAL.

## Contents

| File | Purpose |
|---|---|
| `instances.py` | the 24 problems F1–F12 (GMPB) and P1–P12 (DynPORTAL), each with every generator parameter; 31 runs, 100 environments, fixed seeds |
| `problems.py`, `registration.py` | register F1, …, P12 as problems whose parameters cannot be changed |
| `run.py` | runs one algorithm on all instances and writes the submission files |
| `verify.py`, `protected.sha256` | integrity check of the protected files |
| `algorithms/example_mqso/` | example algorithm: mQSO (multi-swarm PSO) |
| `algorithms/example_mde/` | example algorithm: mDE (multi-population DE) |
| `algorithms/_template/` | skeleton to start from |

## The 24 problems

| Problem | Generator | PeakNumber / ComponentNumber | ChangeFrequency | Dimension | ShiftSeverity |
|---|---|---|---|---|---|
| F1 / P1 | GMPB / DynPORTAL | 5 | 5000 | 5 | 1 |
| F2 / P2 | GMPB / DynPORTAL | 10 | 5000 | 5 | 1 |
| F3 / P3 | GMPB / DynPORTAL | 25 | 5000 | 5 | 1 |
| F4 / P4 | GMPB / DynPORTAL | 50 | 5000 | 5 | 1 |
| F5 / P5 | GMPB / DynPORTAL | 100 | 5000 | 5 | 1 |
| F6 / P6 | GMPB / DynPORTAL | 10 | 2500 | 5 | 1 |
| F7 / P7 | GMPB / DynPORTAL | 10 | 1000 | 5 | 1 |
| F8 / P8 | GMPB / DynPORTAL | 10 | 500 | 5 | 1 |
| F9 / P9 | GMPB / DynPORTAL | 10 | 5000 | 10 | 1 |
| F10 / P10 | GMPB / DynPORTAL | 10 | 5000 | 20 | 1 |
| F11 / P11 | GMPB / DynPORTAL | 10 | 5000 | 5 | 2 |
| F12 / P12 | GMPB / DynPORTAL | 10 | 5000 | 5 | 5 |

Every problem has 100 environments and is run 31 times. `instances.py` lists all other generator parameters of each
problem. Each problem is registered under its name, so it can be used directly, for example
`python -m cli run --algorithm Example_mQSO --problem P8` from the root of the EDOLAB repository (with
`PythonVersion` installed); its parameters cannot be overridden.

## Setting up

Install the dependencies of the Python version once (Python 3.10 or later), then work from the root of the EDOLAB
repository. The competition always uses the sibling `PythonVersion/` folder.

```bash
python -m pip install -e PythonVersion
```

## Implementing an algorithm

1. Copy `algorithms/_template/` to `algorithms/<your_algorithm>/` (no leading underscore) and give the class a unique
   registry name. Every folder in `algorithms/` with an `algorithm.py` is registered automatically.
2. Implement `parameter_specs`, `initialize`, `iterate` and `on_environment_change`. The docstrings of the template and
   of `example_mqso/algorithm.py` list the rules: evaluate only through `evaluate_` (or `problem.evaluate`, which it wraps), return when
   `problem.state.recent_change` or `problem.state.terminated` is set, use `self.rng` for all randomness.
3. Use one parameter setting for all 24 instances: the runner always uses the defaults of `parameter_specs`.

## Running

From the root of the EDOLAB repository:

```bash
python -m CEC2027_Competition.run --algorithm Example_mQSO --smoke                       # quick check
python -m CEC2027_Competition.run --algorithm Example_mQSO --max-parallel-runs 8         # full run
```

The full run performs 24 × 31 runs of 100 environments and writes `CEC2027_Competition/results/<Name>/`:

| File | Content |
|---|---|
| `summary.sqlite`, `trend.sqlite` | the EDOLAB result bundle: one runset per instance (labelled F1 … P12) with the offline error `E_o` and the other metrics of every run, the mean current-error curve of every instance, and the record of the run (seeds, platform, digest of the protected-file manifest) under the meta key `cec2027` |
| `results_table.tex` | best, worst, average, median and standard deviation of `E_o` per instance, in the layout of Table 2 |

The result bundle is the standard output of EDOLAB, so the platform's own tools read it, for example

```bash
python -m cli plot boxplot --result-dir CEC2027_Competition/results/Example_mQSO --plot-metric E_o
python -m cli plot trend --result-dir CEC2027_Competition/results/Example_mQSO --plot-metric current_error
```

from the root of the repository (the figures are written to `PythonVersion/results/plots/`), and the result library of the Relay console imports it to compare algorithms with the
Friedman and Wilcoxon tests. A full run is never overwritten by accident: rerunning it needs `--overwrite`.

`--smoke` (2 runs of 2 environments) and `--instances F1,P3` are for development; their output folders are marked
`_smoke` and `_partial` and are not valid submissions.

## Submission

Compress one folder per algorithm, named after the algorithm, containing:

- the source code of the algorithm (its folder from `algorithms/`),
- the document described in the call for participation (title, participants, description, Table 2),
- `summary.sqlite` and `trend.sqlite` from the output folder.
