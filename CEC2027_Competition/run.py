"""Run an algorithm on the 24 competition instances and write the result files of a submission.

PROTECTED FILE. Do not modify it.

From the root of the EDOLAB repository:

    python -m CEC2027_Competition.run --algorithm <Name> [--max-parallel-runs N] [--output-dir DIR]
    python -m CEC2027_Competition.run --algorithm <Name> --smoke      # quick check, not a valid submission

Every instance is run 31 times with the fixed seeds of instances.py and the algorithm's default parameters. The
output folder ``<output-dir>/<Name>/`` (by default CEC2027_Competition/results/<Name>/) holds:

    summary.sqlite, trend.sqlite   the EDOLAB result bundle: one runset per instance (labelled F1 ... P12) with the
                                   offline error E_o and the other metrics of every run, the mean current-error
                                   curve of every instance, and the record of this run (meta key "cec2027")
    results_table.tex              best, worst, average, median and standard deviation of E_o per instance, in the
                                   layout of Table 2 of the call for participation

The two .sqlite files are the submitted results. They are read by the analysis tools of the platform
(python -m cli sa ..., python -m cli plot ...) and can be imported into the result library of the Relay console,
where the submissions are compared with one another.

Before running, the protected files are checked against protected.sha256; the run is refused when any differs.
"""

from __future__ import annotations

import argparse
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import instances  # noqa: F401  (the platform's algorithms; the platform also imports CEC2027_Competition.algorithms)
from core.experiment.config import BatchExperimentConfig, ExperimentConfig
from core.experiment.runner import ExperimentRunner
from core.registry import algorithm_registry
from core.result.store import SUMMARY_DB_NAME, TREND_DB_NAME, ResultStore
from core.result.validation import validate_result_bundle
from CEC2027_Competition import verify
from CEC2027_Competition.instances import (ALGORITHM_SEEDS, ENVIRONMENT_NUMBER, INSTANCES, PROBLEM_SEEDS,
                                           RUN_NUMBER)

STATISTICS = ("Best", "Worst", "Average", "Median", "Standard deviation")
SMOKE_ENVIRONMENTS = 2
SMOKE_RUNS = 2
META_KEY = "cec2027"
TABLE_NAME = "results_table.tex"
OUTPUT_FILES = (SUMMARY_DB_NAME, TREND_DB_NAME, TABLE_NAME)


def build_tasks(algorithm: str, names: list[str], runs: int, smoke: bool) -> list[ExperimentConfig]:
    tasks = []
    for name in names:
        if smoke:  # the generator with fewer environments; the registered problem cannot be changed
            problem, parameters = INSTANCES[name]
            parameters = {**parameters, "EnvironmentNumber": SMOKE_ENVIRONMENTS}
        else:
            problem, parameters = name, {}
        tasks.append(ExperimentConfig(
            algorithm_names=[algorithm], problem_name=problem, run_number=runs,
            problem_seed=list(PROBLEM_SEEDS[:runs]), algorithm_seed=list(ALGORITHM_SEEDS[:runs]),
            problem_overrides=parameters, label=name,
        ))
    return tasks


def offline_errors(folder: Path) -> dict[str, np.ndarray]:
    """E_o of every run in a result bundle, by instance label and in run order."""
    connection = sqlite3.connect(f"file:{folder / SUMMARY_DB_NAME}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            select runsets.label, runs.run_index, run_metric_values.value
            from run_metric_values
            join runs on runs.id = run_metric_values.run_id
            join runsets on runsets.id = runs.runset_id
            where run_metric_values.metric_name = 'E_o'
            order by runsets.id, runs.run_index
            """
        ).fetchall()
    finally:
        connection.close()
    errors: dict[str, list[float]] = {}
    for label, _, value in rows:
        errors.setdefault(label, []).append(float(value))
    return {label: np.asarray(values) for label, values in errors.items()}


def statistics(values: np.ndarray) -> list[float]:
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return [float(np.min(values)), float(np.max(values)), float(np.mean(values)), float(np.median(values)), std]


def latex_table(names: list[str], table: dict[str, list[float]]) -> str:
    lines = [r"\begin{tabular}{l*{12}{c}}", r"\toprule"]
    for prefix in ("F", "P"):
        block = [name for name in names if name.startswith(prefix)]
        if not block:
            continue
        lines += ["Parameter & " + " & ".join(block) + r"\\", r"\midrule"]
        for index, label in enumerate(STATISTICS):
            lines.append(label + " & " + " & ".join(f"{table[name][index]:.4g}" for name in block) + r"\\")
        lines.append(r"\midrule" if prefix == "F" and len(block) < len(names) else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m CEC2027_Competition.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--algorithm", required=True, help="registry name of the algorithm")
    parser.add_argument("--max-parallel-runs", type=int, default=1, help="number of runs executed in parallel")
    parser.add_argument("--output-dir", type=Path, default=verify.COMPETITION_ROOT / "results")
    parser.add_argument("--instances", default=None,
                        help="comma-separated subset such as F1,P3 (development only; a submission needs all 24)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_RUNS} runs of {SMOKE_ENVIRONMENTS} environments per instance; not a submission")
    parser.add_argument("--overwrite", action="store_true", help="replace the results of an earlier full run")
    args = parser.parse_args(argv)

    problems = verify.check()
    if problems:
        print("Refusing to run: the protected competition files differ from the release.")
        for problem in problems:
            print(f"  {problem}")
        return 1
    if args.algorithm not in algorithm_registry.names():
        print(f"Unknown algorithm {args.algorithm!r}. Registered: {', '.join(algorithm_registry.names())}")
        return 1
    names = list(INSTANCES) if args.instances is None else [item.strip() for item in args.instances.split(",")]
    unknown = [name for name in names if name not in INSTANCES]
    if unknown:
        print(f"Unknown instances {unknown}; use names from {', '.join(INSTANCES)}.")
        return 1

    complete = args.instances is None and not args.smoke
    folder = args.output_dir / (args.algorithm + ("" if complete else "_smoke" if args.smoke else "_partial"))
    existing = [folder / name for name in OUTPUT_FILES if (folder / name).exists()]
    if existing and complete and not args.overwrite:
        print(f"{folder} already holds the results of a full run; pass --overwrite to replace them.")
        return 1
    for path in existing:  # a result bundle is written once; an earlier one in the folder is replaced
        path.unlink()

    runs = SMOKE_RUNS if args.smoke else RUN_NUMBER
    config = BatchExperimentConfig(tasks=build_tasks(args.algorithm, names, runs, args.smoke),
                                   max_parallel_runs=max(1, args.max_parallel_runs), event_sidecar=False)
    started = time.time()

    def progress(result, completed, total):
        print(f"[{completed}/{total}] {result.label} run {result.run_index}: E_o = {result.metrics['E_o']:.6g}",
              flush=True)

    ExperimentRunner().run_batch_to_store(config, folder, progress)

    store = ResultStore(folder)
    try:
        store.set_meta(META_KEY, {
            "algorithm": args.algorithm,
            "valid_submission": complete,
            "instances": names,
            "runs": runs,
            "environment_number": SMOKE_ENVIRONMENTS if args.smoke else ENVIRONMENT_NUMBER,
            "problem_seeds": list(PROBLEM_SEEDS[:runs]),
            "algorithm_seeds": list(ALGORITHM_SEEDS[:runs]),
            "protected_manifest_sha256": verify.manifest_digest(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_time_seconds": round(time.time() - started, 1),
        })
    finally:
        store.close()
    validate_result_bundle(folder, require_completed=True, expected_runs=runs * len(names))

    errors = offline_errors(folder)
    table = {name: statistics(errors[name]) for name in names}
    (folder / TABLE_NAME).write_text(latex_table(names, table), encoding="utf-8")
    print(f"Wrote {folder}" + ("" if complete else " (not a complete submission)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
