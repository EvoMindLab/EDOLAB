"""Visualize EDOLAB OO problem landscapes and NBN snapshots."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import instances  # noqa: F401
from core.parameters import parse_key_value
from core.registry import problem_registry

# results/ of the Python version in a source checkout, ./results in an installed copy (as in cli.py).
_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = _ROOT / "results" if (_ROOT / "pyproject.toml").is_file() else Path("results")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    problem_cls = problem_registry.get(args.problem)
    problem_params = dict(parse_key_value(item) for item in args.problem_param)
    problem = problem_cls.create(problem_params)
    problem.reset(args.problem_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outputs: list[Path] = []

    if args.kind in {"landscape", "both"}:
        landscape_path = output_dir / f"{stamp}_{args.problem}_env{args.environment}_landscape.png"
        problem.landscape_visualize(
            args.environment,
            resolution=args.resolution,
            output_path=landscape_path,
            show=args.show and args.kind == "landscape",
            font_family=args.font_family,
        )
        outputs.append(landscape_path)

    if args.kind in {"nbn", "both"}:
        nbn_path = output_dir / f"{stamp}_{args.problem}_env{args.environment}_nbn.png"
        problem.nbn_visualize(
            args.environment,
            sample_count=args.sample_count,
            seed=args.nbn_seed,
            output_path=nbn_path,
            show=args.show,
            font_family=args.font_family,
        )
        outputs.append(nbn_path)

    for path in outputs:
        print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edolab-oo-visualize-problem")
    parser.add_argument("--problem", default="GMPB")
    parser.add_argument("--problem-seed", type=int, default=1)
    parser.add_argument("--problem-param", action="append", default=[])
    parser.add_argument("--environment", type=int, default=1)
    parser.add_argument("--kind", choices=["landscape", "nbn", "both"], default="both")
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--sample-count", type=int, default=1600)
    parser.add_argument("--nbn-seed", type=int, default=1)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT / "visualizations"))
    parser.add_argument("--show", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
