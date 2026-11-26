"""Command line interface for the EDOLAB OO prototype."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import instances  # noqa: F401
import core.analysis.service as analysis_service
from core.compute import format_compute_report
from core.evaluation.default_metrics import core_metric_output_names
from core.experiment.config import (
    ExperimentConfig,
    load_batch_task_file,
    split_algorithm_overrides,
)
from core.experiment.runner import ExperimentRunner
from core.experiment.seeds import parse_seed_text
from core.parameters import parse_key_value
from core.registry import algorithm_registry, problem_registry
from core.result.plotting import export_boxplot, export_trend_plot
from core.result.store import ResultStore

# Default output folder: results/ next to this file in a source checkout, ./results in an installed copy.
_HERE = Path(__file__).resolve().parent
RESULTS_ROOT = _HERE / "results" if (_HERE / "pyproject.toml").is_file() else Path("results")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        print("Algorithms:", ", ".join(algorithm_registry.names()))
        print("Problems:", ", ".join(problem_registry.names()))
        print("Metrics:", ", ".join(list_metric_output_names()))
        return 0
    if args.command == "run":
        return run_command(args)
    if args.command == "batch":
        return batch_command(args)
    if args.command == "sa":
        return sa_command(args)
    if args.command == "plot":
        return plot_command(args)
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edolab-oo")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List registered instances.")

    run = subparsers.add_parser("run", help="Run experiments.")
    run.add_argument("--algorithm", action="append", dest="algorithms", required=True)
    run.add_argument("--problem", default="GMPB")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--problem-seed", default=None)
    run.add_argument("--algorithm-seed", default=None)
    run.add_argument("--max-parallel-runs", type=int, default=None)
    run.add_argument("--problem-param", action="append", default=[])
    run.add_argument("--algorithm-param", action="append", default=[])
    run.add_argument(
        "--metric-output",
        dest="metric_outputs",
        action="append",
        default=[],
        help=(
            "Comma-separated scalar or trend metrics to store/export; "
            "omitted means store/export all computed metrics."
        ),
    )
    run.add_argument("--output-dir", default=None)

    batch = subparsers.add_parser(
        "batch", help="Run experiments from a JSON Lines task file."
    )
    batch.add_argument("--task-file", required=True)

    sa = subparsers.add_parser("sa", help="Analyze stored experiment results.")
    sa_subparsers = sa.add_subparsers(dest="sa_command")
    sa_list = sa_subparsers.add_parser(
        "list-groups", help="List statistically compatible runset groups."
    )
    sa_list.add_argument("--result-dir", required=True)
    sa_run = sa_subparsers.add_parser("run", help="Run statistical analysis.")
    sa_run.add_argument("--result-dir", required=True)
    sa_run.add_argument(
        "--method",
        required=True,
        choices=("friedman", "signed-rank", "rank-sum"),
    )
    sa_run.add_argument(
        "--sa-metric",
        action="append",
        default=[],
        help="Scalar metric to analyze; repeat or omit to analyze all compatible scalar metrics.",
    )
    sa_run.add_argument(
        "--sa-metrics",
        default=None,
        help="Comma-separated scalar metrics to analyze; omitted means all compatible scalar metrics.",
    )
    sa_run.add_argument("--runset-id", action="append", type=int, default=[])
    sa_run.add_argument("--runset-uid", action="append", default=[])
    sa_run.add_argument("--alpha", type=float, default=0.05)
    sa_run.add_argument(
        "--format",
        dest="formats",
        action="append",
        default=[],
        help="Comma-separated output formats: sqlite,csv,xlsx,latex.",
    )
    sa_run.add_argument("--output-dir", default=None)

    plot = subparsers.add_parser("plot", help="Export plots from stored results.")
    plot_subparsers = plot.add_subparsers(dest="plot_command")
    plot_trend = plot_subparsers.add_parser("trend", help="Export aggregate trend plot.")
    plot_trend.add_argument("--result-dir", required=True)
    plot_trend.add_argument("--runset-uid", action="append", default=[])
    plot_trend.add_argument("--runset-id", action="append", type=int, default=[])
    plot_trend.add_argument("--plot-metric", required=True)
    plot_trend.add_argument("--format", dest="formats", action="append", default=[])
    plot_trend.add_argument("--output-dir", default=None)
    plot_box = plot_subparsers.add_parser("boxplot", help="Export scalar metric boxplot.")
    plot_box.add_argument("--result-dir", required=True)
    plot_box.add_argument("--runset-uid", action="append", default=[])
    plot_box.add_argument("--runset-id", action="append", type=int, default=[])
    plot_box.add_argument("--plot-metric", required=True)
    plot_box.add_argument("--format", dest="formats", action="append", default=[])
    plot_box.add_argument("--output-dir", default=None)
    return parser


def run_command(args: argparse.Namespace) -> int:
    algorithms = args.algorithms
    problem_overrides = dict(parse_key_value(item) for item in args.problem_param)
    shared_algorithm_overrides, algorithm_overrides = parse_cli_algorithm_overrides(
        args.algorithm_param,
        algorithms,
    )
    output_dir = str(
        normalize_raw_run_output_dir(args.output_dir)
        if args.output_dir
        else RESULTS_ROOT
        / "raw_runs"
        / time.strftime("edolab_python_oo_run_%Y%m%d_%H%M%S")
    )
    config = ExperimentConfig(
        algorithm_names=algorithms,
        problem_name=args.problem,
        run_number=args.runs,
        problem_seed=parse_seed_text(args.problem_seed),
        algorithm_seed=parse_seed_text(args.algorithm_seed),
        problem_overrides=problem_overrides,
        shared_algorithm_overrides=shared_algorithm_overrides,
        algorithm_overrides=algorithm_overrides,
        metric_output_names=parse_metric_output_names(args.metric_outputs) or None,
        max_parallel_runs=args.max_parallel_runs or (__import__("os").cpu_count() or 1),
        output_dir=output_dir,
    )

    total = len(config.algorithm_names) * config.run_number
    print(
        f"Starting {total} run(s): algorithms={','.join(config.algorithm_names)}, "
        f"problem={config.problem_name}, max_parallel_runs={config.max_parallel_runs}"
    )

    completed_by_algorithm = {name: 0 for name in config.algorithm_names}

    def progress(result, completed: int, total_runs: int) -> None:
        completed_by_algorithm[result.algorithm_name] += 1
        metrics = filter_metric_values(result.metrics, config.metric_output_names)
        print(
            f"[{completed}/{total_runs}] {result.algorithm_name} run {result.run_index} "
            f"complete: FE={result.fe}, runtime={result.runtime_seconds:.3f}s, "
            f"{format_metric_values(metrics)}, "
            f"algorithm_progress={completed_by_algorithm[result.algorithm_name]}/{config.run_number}, "
            f"{format_compute_report(result.raw.get('compute'))}"
        )

    try:
        output_path = ExperimentRunner().run_to_store(
            config,
            output_dir,
            progress=progress,
        )
    except (NotImplementedError, ValueError, KeyError) as exc:
        print(f"Run aborted: {exc}")
        return 2

    print(f"Wrote raw run results to {output_path}")
    print(f"Wrote {Path(output_path) / 'summary.sqlite'}")
    print(f"Wrote {Path(output_path) / 'trend.sqlite'}")
    return 0


def batch_command(args: argparse.Namespace) -> int:
    try:
        config = load_batch_task_file(args.task_file)
    except (OSError, ValueError) as exc:
        print(f"Batch aborted: {exc}")
        return 2

    output_dir = timestamped_batch_output_dir(
        normalize_raw_run_output_dir(
            config.output_dir
            or str(RESULTS_ROOT / "raw_runs" / "edolab_python_oo_batch")
        )
    )
    config.output_dir = output_dir
    for task in config.tasks:
        task.output_dir = output_dir

    total = sum(len(task.algorithm_names) * task.run_number for task in config.tasks)
    print(
        f"Starting {total} run(s) from {len(config.tasks)} task(s): "
        f"max_parallel_runs={config.max_parallel_runs}"
    )

    expected_by_task = {
        (task.label, algorithm_name): task.run_number
        for task in config.tasks
        for algorithm_name in task.algorithm_names
    }
    completed_by_task = {key: 0 for key in expected_by_task}
    metric_outputs_by_task = {
        (task.label, algorithm_name): task.metric_output_names
        for task in config.tasks
        for algorithm_name in task.algorithm_names
    }

    def progress(result, completed: int, total_runs: int) -> None:
        key = (result.label, result.algorithm_name)
        completed_by_task[key] += 1
        label_text = f"{result.label}: " if result.label else ""
        metrics = filter_metric_values(
            result.metrics,
            metric_outputs_by_task.get(key),
        )
        print(
            f"[{completed}/{total_runs}] {label_text}{result.algorithm_name} "
            f"run {result.run_index} complete: FE={result.fe}, "
            f"runtime={result.runtime_seconds:.3f}s, "
            f"{format_metric_values(metrics)}, "
            f"task_progress={completed_by_task[key]}/{expected_by_task[key]}, "
            f"{format_compute_report(result.raw.get('compute'))}"
        )

    try:
        output_path = ExperimentRunner().run_batch_to_store(
            config,
            output_dir,
            progress=progress,
        )
    except (NotImplementedError, ValueError, KeyError) as exc:
        print(f"Batch aborted: {exc}")
        return 2

    print(f"Wrote raw run results to {output_path}")
    print(f"Wrote {Path(output_path) / 'summary.sqlite'}")
    print(f"Wrote {Path(output_path) / 'trend.sqlite'}")
    return 0


def sa_command(args: argparse.Namespace) -> int:
    if args.sa_command == "list-groups":
        return sa_list_groups_command(args)
    if args.sa_command == "run":
        return sa_run_command(args)
    print("SA aborted: choose one of list-groups or run.")
    return 2


def sa_list_groups_command(args: argparse.Namespace) -> int:
    try:
        store = open_result_store(args.result_dir)
    except ValueError as exc:
        print(f"SA aborted: {exc}")
        return 2
    try:
        groups = analysis_service.compatible_runset_groups(store)
        if not groups:
            print("No compatible runset groups found.")
            return 0
        for index, group in enumerate(groups, start=1):
            metrics = analysis_service.common_scalar_metrics(
                store,
                [item["runset_id"] for item in group],
            )
            first = group[0]
            params = json.dumps(first["problem_params"], sort_keys=True)
            print(
                f"Group {index}: problem={first['problem_name']}, "
                f"runsets={len(group)}, metrics={','.join(metrics) or '<none>'}"
            )
            print(f"  problem_params={params}")
            for item in group:
                print(
                    f"  {item['runset_id']}: {item['label']} / "
                    f"{item['algorithm_name']} uid={item['runset_uid']} "
                    f"({item['completed_runs']}/{item['expected_runs']} runs)"
                )
        return 0
    finally:
        store.close()


def sa_run_command(args: argparse.Namespace) -> int:
    try:
        store = open_result_store(args.result_dir)
    except ValueError as exc:
        print(f"SA aborted: {exc}")
        return 2
    try:
        try:
            metric_names = parse_sa_metric_names(
                store,
                args.sa_metric,
                args.sa_metrics,
                args.method,
                args.runset_id,
                args.runset_uid,
            )
            payload = analysis_service.run_statistical_analysis(
                store,
                method=args.method,
                metrics=metric_names,
                runset_ids=args.runset_id,
                runset_uids=args.runset_uid,
                alpha=args.alpha,
            )
        except ValueError as exc:
            print(f"SA aborted: {exc}")
            return 2

        output_dir = args.output_dir or str(
            RESULTS_ROOT
            / "statistical_analysis"
            / time.strftime(f"{args.method}_{'_'.join(payload['metrics'])}_%Y%m%d_%H%M%S")
        )
        written = analysis_service.write_analysis_outputs(
            output_dir,
            payload,
            parse_output_formats(args.formats) or ["sqlite"],
        )
        print(f"Wrote statistical analysis to {output_dir}")
        for key, path in written.items():
            print(f"Wrote {key}: {path}")
        return 0
    finally:
        store.close()


def plot_command(args: argparse.Namespace) -> int:
    if args.plot_command == "trend":
        return plot_trend_command(args)
    if args.plot_command == "boxplot":
        return plot_boxplot_command(args)
    print("Plot aborted: choose one of trend or boxplot.")
    return 2


def plot_trend_command(args: argparse.Namespace) -> int:
    try:
        runset_uids = resolve_cli_runset_uids(args.result_dir, args.runset_id, args.runset_uid)
        output_dir = args.output_dir or str(
            RESULTS_ROOT
            / "plots"
            / time.strftime(f"trend_{args.plot_metric}_%Y%m%d_%H%M%S")
        )
        written = export_trend_plot(
            args.result_dir,
            runset_uids=runset_uids,
            metric_name=args.plot_metric,
            output_dir=output_dir,
            formats=parse_output_formats(args.formats) or ["pdf"],
        )
    except ValueError as exc:
        print(f"Plot aborted: {exc}")
        return 2
    for fmt, path in written.items():
        print(f"Wrote {fmt}: {path}")
    return 0


def plot_boxplot_command(args: argparse.Namespace) -> int:
    try:
        runset_uids = resolve_cli_runset_uids(args.result_dir, args.runset_id, args.runset_uid)
        output_dir = args.output_dir or str(
            RESULTS_ROOT
            / "plots"
            / time.strftime(f"boxplot_{args.plot_metric}_%Y%m%d_%H%M%S")
        )
        written = export_boxplot(
            args.result_dir,
            runset_uids=runset_uids,
            metric_name=args.plot_metric,
            output_dir=output_dir,
            formats=parse_output_formats(args.formats) or ["pdf"],
        )
    except ValueError as exc:
        print(f"Plot aborted: {exc}")
        return 2
    for fmt, path in written.items():
        print(f"Wrote {fmt}: {path}")
    return 0


def open_result_store(result_dir: str | Path) -> ResultStore:
    path = Path(result_dir)
    if not (path / "summary.sqlite").exists():
        raise ValueError(f"Missing {path / 'summary.sqlite'}.")
    if not (path / "trend.sqlite").exists():
        raise ValueError(f"Missing {path / 'trend.sqlite'}.")
    return ResultStore(path)


def parse_cli_algorithm_overrides(
    items: list[str],
    algorithms: list[str],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    overrides: dict[str, object] = {}
    for item in items:
        key, value = parse_key_value(item)
        overrides[key] = value
    return split_algorithm_overrides(overrides, algorithms)


def parse_algorithm_overrides(
    items: list[str],
    algorithms: list[str],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    return parse_cli_algorithm_overrides(items, algorithms)


def parse_metric_output_names(items: list[str]) -> list[str]:
    names: list[str] = []
    for item in items:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    return names


def parse_sa_metric_names(
    store: ResultStore,
    sa_metric: list[str],
    sa_metrics: str | None,
    method: str,
    runset_ids: list[int],
    runset_uids: list[str],
) -> list[str]:
    names: list[str] = []
    if sa_metrics:
        names.extend(part.strip() for part in sa_metrics.split(",") if part.strip())
    for item in sa_metric:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    unique: list[str] = []
    for name in names:
        if name and name not in unique:
            unique.append(name)
    if unique:
        return unique

    if runset_uids:
        selected_ids = store.resolve_runset_ids(runset_uids)
    elif runset_ids:
        selected_ids = runset_ids
    else:
        minimum = 3 if method == "friedman" else 2
        selected_ids = []
        for group in analysis_service.compatible_runset_groups(store):
            ids = [item["runset_id"] for item in group]
            metrics = analysis_service.common_scalar_metrics(store, ids)
            usable = [
                name
                for name in metrics
                if _metric_direction_supported(store, ids, name)
            ]
            if len(ids) >= minimum and usable:
                selected_ids = ids
                return usable
        raise ValueError("No compatible scalar metrics found for SA.")
    metrics = analysis_service.common_scalar_metrics(store, list(selected_ids))
    usable = [
        name for name in metrics if _metric_direction_supported(store, list(selected_ids), name)
    ]
    if not usable:
        raise ValueError("No compatible scalar metrics found for SA.")
    return usable


def parse_output_formats(items: list[str]) -> list[str]:
    formats: list[str] = []
    for item in items:
        formats.extend(part.strip() for part in item.split(",") if part.strip())
    return formats


def resolve_cli_runset_uids(
    result_dir: str | Path,
    runset_ids: list[int],
    runset_uids: list[str],
) -> list[str]:
    if runset_uids:
        return runset_uids
    store = open_result_store(result_dir)
    try:
        runsets = store.list_runsets()
        if runset_ids:
            by_id = {item["runset_id"]: item["runset_uid"] for item in runsets}
            missing = [runset_id for runset_id in runset_ids if runset_id not in by_id]
            if missing:
                raise ValueError(f"Unknown runset id(s): {', '.join(map(str, missing))}.")
            return [by_id[runset_id] for runset_id in runset_ids]
        return [item["runset_uid"] for item in runsets]
    finally:
        store.close()


def _metric_direction_supported(
    store: ResultStore,
    runset_ids: list[int],
    metric_name: str,
) -> bool:
    samples = store.load_metric_samples(runset_ids, metric_name)
    return bool(samples) and all(sample.direction in {"lower", "higher"} for sample in samples)


def format_metric_values(metrics: dict[str, float]) -> str:
    if not metrics:
        return "metrics=<none>"
    return ", ".join(f"{name}={value:.6g}" for name, value in metrics.items())


def filter_metric_values(
    metrics: dict[str, float],
    metric_output_names: list[str] | tuple[str, ...] | None,
) -> dict[str, float]:
    if metric_output_names is None:
        return metrics
    allowed = set(metric_output_names)
    return {name: value for name, value in metrics.items() if name in allowed}


def list_metric_output_names() -> list[str]:
    names = set(core_metric_output_names())
    for algorithm_name in algorithm_registry.names():
        algorithm_cls = algorithm_registry.get(algorithm_name)
        names.update(
            metric_cls.name for metric_cls in algorithm_cls.custom_metric_classes()
        )
    return sorted(names)


def timestamped_batch_output_dir(output_dir: str | Path) -> str:
    """Append a wall-clock timestamp suffix to a batch output directory."""

    path = Path(output_dir)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    if path.name:
        return str(path.with_name(f"{path.name}_{timestamp}"))
    return str(path / timestamp)


def normalize_raw_run_output_dir(output_dir: str | Path) -> Path:
    """A relative path that starts with results/ is placed in RESULTS_ROOT, runs under raw_runs/."""

    path = Path(output_dir)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "results":
        if len(parts) < 2 or parts[1] != "raw_runs":
            name = path.name if path.name != "results" else "edolab_python_oo"
            return RESULTS_ROOT / "raw_runs" / name
        return RESULTS_ROOT.joinpath(*parts[1:])
    return path


if __name__ == "__main__":
    raise SystemExit(main())
