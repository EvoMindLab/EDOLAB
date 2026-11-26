"""Safe plot-export boundary for the Relay application.

It accepts result series already loaded from verified local bundles, validates
and bounds them, delegates the numerical presentation to
``core.result.plotting``, and publishes only files created inside its private
artifact workspace.  Interactive requests call it directly in the local Relay
process; it does not know about execution targets, queues, or SSH.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import io
import math
from pathlib import Path
import shutil
import threading
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from core.result import plotting


PLOT_FORMATS = frozenset({"pdf", "svg", "png"})
TREND_LEGEND_POSITIONS = frozenset(
    {
        "outside_right",
        "bestoutside",
        "best",
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "northwest",
        "northeast",
        "southwest",
        "southeast",
    }
)
Y_SCALES = frozenset({"auto", "linear", "log"})

DEFAULT_MAX_SERIES = 80
DEFAULT_MAX_POINTS_PER_SERIES = 20_000
DEFAULT_MAX_TOTAL_POINTS = 200_000
DEFAULT_MAX_ARTIFACT_DIRECTORIES = 128
MAX_METRIC_NAME_LENGTH = 256
MAX_TITLE_LENGTH = 2_048
MAX_LABEL_LENGTH = 512
MAX_AXIS_LENGTH = 128
MAX_KEY_TEXT_LENGTH = 512

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
    "png": "image/png",
}


class PlotWorkspaceError(ValueError):
    """Raised when a plot request or managed artifact is invalid."""


@dataclass(frozen=True, slots=True)
class TrendPlotRequest:
    """Canonical local interactive trend request."""

    metric_name: str
    series: tuple[plotting.TrendPlotSeries, ...]
    formats: tuple[str, ...]
    title: str | None
    x_min: float | None
    x_max: float | None
    legend_position: str
    y_scale: str


@dataclass(frozen=True, slots=True)
class BoxplotRequest:
    """Canonical local interactive scalar-sample request."""

    metric_name: str
    series: tuple[plotting.BoxplotSeries, ...]
    formats: tuple[str, ...]
    title: str | None


@dataclass(frozen=True, slots=True)
class PlotPreview:
    """One PNG rendered entirely in memory for an interactive preview."""

    content: bytes
    kind: str
    metric_name: str
    series_count: int
    point_count: int
    axis: str | None = None


class PlotWorkspace:
    """Validate result series and expose verified plot artifacts.

    Callers never choose an output path or filename.  Every export receives a
    fresh UUID directory below ``artifact_root`` and a failed export removes
    that directory in full.
    """

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        artifact_url_prefix: str = "/api/plots/artifacts",
        max_series: int = DEFAULT_MAX_SERIES,
        max_points_per_series: int = DEFAULT_MAX_POINTS_PER_SERIES,
        max_total_points: int = DEFAULT_MAX_TOTAL_POINTS,
        max_artifact_directories: int = DEFAULT_MAX_ARTIFACT_DIRECTORIES,
    ) -> None:
        root = Path(artifact_root).expanduser()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink() or not root.is_dir():
            raise PlotWorkspaceError("artifact_root must be a regular directory, not a symlink.")
        self.artifact_root = root.resolve(strict=True)
        self.artifact_url_prefix = "/" + artifact_url_prefix.strip("/")
        self.max_series = _strict_positive_int(max_series, "max_series")
        self.max_points_per_series = _strict_positive_int(
            max_points_per_series, "max_points_per_series"
        )
        self.max_total_points = _strict_positive_int(
            max_total_points, "max_total_points"
        )
        self.max_artifact_directories = _strict_positive_int(
            max_artifact_directories, "max_artifact_directories"
        )
        if self.max_artifact_directories > 10_000:
            raise PlotWorkspaceError(
                "max_artifact_directories must be at most 10000."
            )
        self._artifact_lock = threading.Lock()
        if self.max_total_points < self.max_points_per_series:
            raise PlotWorkspaceError(
                "max_total_points must be at least max_points_per_series."
            )

    def limits(self) -> dict[str, Any]:
        """Return the public validation limits used by the plot form."""

        return {
            "formats": sorted(PLOT_FORMATS),
            "legend_positions": sorted(TREND_LEGEND_POSITIONS),
            "y_scales": sorted(Y_SCALES),
            "max_series": self.max_series,
            "max_points_per_series": self.max_points_per_series,
            "max_total_points": self.max_total_points,
        }

    def validate_request(
        self,
        payload: Mapping[str, Any],
        *,
        operation: str,
    ) -> TrendPlotRequest | BoxplotRequest:
        """Return a canonical request without creating any output files."""

        normalized_operation = _normalize_operation(operation)
        if normalized_operation == "trend":
            return self._validate_trend_request(payload)
        return self._validate_boxplot_request(payload)

    def export_trend(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Export a trend plot into a fresh managed directory."""

        request = self._validate_trend_request(payload)
        run_dir = self._new_artifact_directory()
        try:
            written = plotting.export_trend_series(
                request.series,
                metric_name=request.metric_name,
                output_dir=run_dir,
                formats=request.formats,
                title=request.title,
                x_min=request.x_min,
                x_max=request.x_max,
                legend_position=request.legend_position,
                y_scale=request.y_scale,
            )
            artifacts = self._verify_export(
                written,
                requested_formats=request.formats,
                run_dir=run_dir,
            )
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return self._result_payload(
            kind="trend_plot",
            metric_name=request.metric_name,
            series_count=len(request.series),
            point_count=sum(len(item.key) for item in request.series),
            artifacts=artifacts,
            axis=request.series[0].axis,
        )

    def export_plot(
        self,
        payload: Mapping[str, Any],
        plot_type: str,
    ) -> dict[str, Any]:
        """Dispatch a worker/API plot type through the same validated boundary."""

        operation = _normalize_operation(plot_type)
        if operation == "trend":
            return self.export_trend(payload)
        return self.export_boxplot(payload)

    def preview_plot(
        self,
        payload: Mapping[str, Any],
        plot_type: str,
    ) -> PlotPreview:
        """Render one PNG without creating a managed artifact.

        Preview requests share the same canonical validation and point budgets
        as exports, but Matplotlib writes directly into a ``BytesIO`` buffer.
        No request directory, storage key, or downloadable artifact is created.
        """

        operation = _normalize_operation(plot_type)
        request = self.validate_request(payload, operation=operation)
        if isinstance(request, TrendPlotRequest):
            content = _render_trend_png(request)
            return PlotPreview(
                content=content,
                kind="trend_plot",
                metric_name=request.metric_name,
                series_count=len(request.series),
                point_count=sum(len(item.key) for item in request.series),
                axis=request.series[0].axis,
            )
        content = _render_boxplot_png(request)
        return PlotPreview(
            content=content,
            kind="boxplot",
            metric_name=request.metric_name,
            series_count=len(request.series),
            point_count=sum(len(item.values) for item in request.series),
        )

    def export_boxplot(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Export a scalar-sample boxplot into a fresh managed directory."""

        request = self._validate_boxplot_request(payload)
        run_dir = self._new_artifact_directory()
        try:
            written = plotting.export_boxplot_series(
                request.series,
                metric_name=request.metric_name,
                output_dir=run_dir,
                formats=request.formats,
                title=request.title,
            )
            artifacts = self._verify_export(
                written,
                requested_formats=request.formats,
                run_dir=run_dir,
            )
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        return self._result_payload(
            kind="boxplot",
            metric_name=request.metric_name,
            series_count=len(request.series),
            point_count=sum(len(item.values) for item in request.series),
            artifacts=artifacts,
        )

    # Naming aliases make the boundary easy to use from either a synchronous
    # HTTP handler (``render_*``) or a worker action (``export_*``).
    render_trend = export_trend
    render_boxplot = export_boxplot

    def resolve_artifact(self, storage_key: str) -> Path:
        """Resolve one plot storage key without allowing traversal or symlinks."""

        if not isinstance(storage_key, str) or not storage_key.strip():
            raise PlotWorkspaceError("artifact storage_key must be a non-empty string.")
        if "\x00" in storage_key or "\\" in storage_key:
            raise PlotWorkspaceError("artifact storage_key contains invalid characters.")
        relative = Path(storage_key)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise PlotWorkspaceError(
                "artifact storage_key must be a safe relative path."
            )
        if (
            len(relative.parts) != 2
            or len(relative.parts[0]) != 32
            or any(character not in "0123456789abcdef" for character in relative.parts[0])
        ):
            raise PlotWorkspaceError(
                "artifact storage_key is not a managed plot key."
            )
        fmt = relative.suffix.lower().removeprefix(".")
        if fmt not in PLOT_FORMATS:
            raise PlotWorkspaceError(
                "artifact storage_key does not identify a supported plot file."
            )
        cursor = self.artifact_root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise PlotWorkspaceError(
                    "artifact storage_key must not traverse symlinks."
                )
        try:
            resolved = (self.artifact_root / relative).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Plot artifact does not exist: {storage_key}") from exc
        if not resolved.is_relative_to(self.artifact_root):
            raise PlotWorkspaceError("artifact storage_key escapes the managed workspace.")
        if not resolved.is_file():
            raise PlotWorkspaceError("artifact storage_key is not a regular file.")
        _validate_file_signature(resolved, fmt)
        return resolved

    def _validate_trend_request(self, payload: Mapping[str, Any]) -> TrendPlotRequest:
        mapping = _require_payload(payload)
        allowed = {
            "metric_name",
            "series",
            "formats",
            "title",
            "x_min",
            "x_max",
            "legend_position",
            "y_scale",
        }
        _reject_unknown_fields(mapping, allowed)
        metric_name = _bounded_text(
            mapping.get("metric_name"),
            "metric_name",
            maximum=MAX_METRIC_NAME_LENGTH,
        )
        title = _optional_text(mapping.get("title"), "title", maximum=MAX_TITLE_LENGTH)
        formats = _validate_formats(mapping.get("formats"))
        legend_position = _choice(
            mapping.get("legend_position", "outside_right"),
            "legend_position",
            TREND_LEGEND_POSITIONS,
        )
        y_scale = _choice(mapping.get("y_scale", "auto"), "y_scale", Y_SCALES)
        x_min = _optional_finite_number(mapping.get("x_min"), "x_min")
        x_max = _optional_finite_number(mapping.get("x_max"), "x_max")
        if x_min is not None and x_max is not None and x_min > x_max:
            raise PlotWorkspaceError("x_min must be less than or equal to x_max.")

        raw_series = self._validate_series_container(mapping.get("series"))
        parsed: list[plotting.TrendPlotSeries] = []
        seen_labels: set[str] = set()
        common_axis: str | None = None
        for index, item in enumerate(raw_series):
            entry = _require_series_mapping(item, index)
            _reject_unknown_fields(
                entry,
                {"label", "axis", "key", "mean", "stderr", "count"},
                prefix=f"series[{index}] ",
            )
            label = _bounded_text(
                entry.get("label"),
                f"series[{index}].label",
                maximum=MAX_LABEL_LENGTH,
            )
            if label in seen_labels:
                raise PlotWorkspaceError(f"Duplicate series label: {label!r}.")
            seen_labels.add(label)
            axis = _bounded_text(
                entry.get("axis"),
                f"series[{index}].axis",
                maximum=MAX_AXIS_LENGTH,
            )
            if common_axis is None:
                common_axis = axis
            elif axis != common_axis:
                raise PlotWorkspaceError(
                    f"Trend series must share one axis; received {common_axis!r} and {axis!r}."
                )
            keys = _key_list(entry.get("key"), f"series[{index}].key")
            if (x_min is not None or x_max is not None) and any(
                not _is_finite_numeric_key(key) for key in keys
            ):
                raise PlotWorkspaceError(
                    "x_min and x_max can only be used with numeric trend keys."
                )
            means = _finite_number_list(entry.get("mean"), f"series[{index}].mean")
            if len(keys) != len(means):
                raise PlotWorkspaceError(
                    f"series[{index}].key and mean must have the same length."
                )
            stderr = _optional_number_list(
                entry.get("stderr"),
                f"series[{index}].stderr",
                expected_length=len(keys),
                minimum=0.0,
            )
            count = _optional_count_list(
                entry.get("count"),
                f"series[{index}].count",
                expected_length=len(keys),
            )
            point_count = len(keys)
            if point_count == 0:
                raise PlotWorkspaceError(
                    f"series[{index}] must contain at least one point."
                )
            parsed.append(
                plotting.TrendPlotSeries(
                    label=label,
                    axis=axis,
                    key=keys,
                    mean=means,
                    stderr=stderr,
                    count=count,
                )
            )
        if y_scale == "log" and any(
            value <= 0 for item in parsed for value in item.mean
        ):
            raise PlotWorkspaceError(
                "y_scale='log' requires every trend mean to be greater than zero."
            )
        try:
            parsed = plotting.downsample_trend_series_collection(
                parsed,
                max_points_per_series=self.max_points_per_series,
                max_total_points=self.max_total_points,
            )
        except ValueError as exc:
            raise PlotWorkspaceError(str(exc)) from exc
        return TrendPlotRequest(
            metric_name=metric_name,
            series=tuple(parsed),
            formats=formats,
            title=title,
            x_min=x_min,
            x_max=x_max,
            legend_position=legend_position,
            y_scale=y_scale,
        )

    def _validate_boxplot_request(self, payload: Mapping[str, Any]) -> BoxplotRequest:
        mapping = _require_payload(payload)
        _reject_unknown_fields(mapping, {"metric_name", "series", "formats", "title"})
        metric_name = _bounded_text(
            mapping.get("metric_name"),
            "metric_name",
            maximum=MAX_METRIC_NAME_LENGTH,
        )
        title = _optional_text(mapping.get("title"), "title", maximum=MAX_TITLE_LENGTH)
        formats = _validate_formats(mapping.get("formats"))
        raw_series = self._validate_series_container(mapping.get("series"))
        parsed: list[plotting.BoxplotSeries] = []
        seen_labels: set[str] = set()
        total_points = 0
        for index, item in enumerate(raw_series):
            entry = _require_series_mapping(item, index)
            _reject_unknown_fields(
                entry,
                {"label", "values"},
                prefix=f"series[{index}] ",
            )
            label = _bounded_text(
                entry.get("label"),
                f"series[{index}].label",
                maximum=MAX_LABEL_LENGTH,
            )
            if label in seen_labels:
                raise PlotWorkspaceError(f"Duplicate series label: {label!r}.")
            seen_labels.add(label)
            values = _finite_number_list(
                entry.get("values"), f"series[{index}].values"
            )
            point_count = len(values)
            self._check_point_limit(point_count, index=index)
            total_points += point_count
            if total_points > self.max_total_points:
                raise PlotWorkspaceError(
                    f"Plot has {total_points} points; maximum is {self.max_total_points}."
                )
            parsed.append(plotting.BoxplotSeries(label=label, values=values))
        return BoxplotRequest(
            metric_name=metric_name,
            series=tuple(parsed),
            formats=formats,
            title=title,
        )

    def _validate_series_container(self, value: Any) -> list[Any] | tuple[Any, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise PlotWorkspaceError("series must be a non-empty array.")
        if len(value) > self.max_series:
            raise PlotWorkspaceError(
                f"Plot has {len(value)} series; maximum is {self.max_series}."
            )
        return value

    def _check_point_limit(self, point_count: int, *, index: int) -> None:
        if point_count == 0:
            raise PlotWorkspaceError(f"series[{index}] must contain at least one point.")
        if point_count > self.max_points_per_series:
            raise PlotWorkspaceError(
                f"series[{index}] has {point_count} points; maximum is "
                f"{self.max_points_per_series}."
            )

    def _new_artifact_directory(self) -> Path:
        with self._artifact_lock:
            self._prune_artifact_directories()
            run_dir = self.artifact_root / uuid4().hex
            if run_dir.parent != self.artifact_root:
                raise PlotWorkspaceError("Generated artifact directory escaped the workspace.")
            run_dir.mkdir(mode=0o700)
        return run_dir

    def _prune_artifact_directories(self) -> None:
        """Bound transient interactive exports without creating run records."""

        managed = sorted(
            (
                item
                for item in self.artifact_root.iterdir()
                if not item.is_symlink()
                and item.is_dir()
                and len(item.name) == 32
                and all(character in "0123456789abcdef" for character in item.name)
            ),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        keep_before_create = max(self.max_artifact_directories - 1, 0)
        for stale in managed[keep_before_create:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _verify_export(
        self,
        written: Mapping[str, str | Path],
        *,
        requested_formats: tuple[str, ...],
        run_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(written, Mapping):
            raise PlotWorkspaceError("Plot exporter returned invalid artifact metadata.")
        returned_formats = set(written)
        expected_formats = set(requested_formats)
        if returned_formats != expected_formats:
            raise PlotWorkspaceError(
                "Plot exporter did not return exactly the requested formats."
            )
        artifacts: dict[str, dict[str, Any]] = {}
        managed_files: set[Path] = set()
        created_at = datetime.now(UTC).isoformat()
        for fmt in requested_formats:
            raw_target = written[fmt]
            if not isinstance(raw_target, (str, Path)):
                raise PlotWorkspaceError("Plot exporter returned an invalid artifact path.")
            target = Path(raw_target)
            if target.is_symlink():
                raise PlotWorkspaceError("Plot exporter returned a symlink artifact.")
            try:
                resolved = target.resolve(strict=True)
            except FileNotFoundError as exc:
                raise PlotWorkspaceError(
                    f"Plot exporter did not create the requested {fmt.upper()} artifact."
                ) from exc
            if resolved.parent != run_dir or not resolved.is_relative_to(self.artifact_root):
                raise PlotWorkspaceError(
                    "Plot exporter returned a file outside its managed request directory."
                )
            if not resolved.is_file() or resolved.suffix.lower() != f".{fmt}":
                raise PlotWorkspaceError(
                    f"Plot exporter returned an invalid {fmt.upper()} artifact."
                )
            size = resolved.stat().st_size
            if size <= 0:
                raise PlotWorkspaceError("Plot exporter returned an empty artifact.")
            _validate_file_signature(resolved, fmt)
            managed_files.add(resolved)
            storage_key = resolved.relative_to(self.artifact_root).as_posix()
            artifacts[fmt] = {
                "id": f"{run_dir.name}:{fmt}",
                "storage_key": storage_key,
                "filename": resolved.name,
                "format": fmt,
                "media_type": _MEDIA_TYPES[fmt],
                "size_bytes": size,
                "sha256": _sha256_file(resolved),
                "url": f"{self.artifact_url_prefix}/{quote(storage_key, safe='/')}",
                "created_at": created_at,
            }
        actual_files = set(run_dir.iterdir())
        if any(item.is_symlink() for item in actual_files) or actual_files != managed_files:
            raise PlotWorkspaceError(
                "Plot exporter created unexpected files in its managed request directory."
            )
        return artifacts

    @staticmethod
    def _result_payload(
        *,
        kind: str,
        metric_name: str,
        series_count: int,
        point_count: int,
        artifacts: dict[str, dict[str, Any]],
        axis: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": kind,
            "metric_name": metric_name,
            "series_count": series_count,
            "point_count": point_count,
            "artifacts": artifacts,
            "preview_artifact": artifacts.get("png"),
        }
        if axis is not None:
            result["axis"] = axis
        return result


def _normalize_operation(value: str) -> str:
    if not isinstance(value, str):
        raise PlotWorkspaceError("Plot operation must be 'trend' or 'boxplot'.")
    normalized = value.strip().lower()
    aliases = {"trend_plot": "trend", "box": "boxplot", "box_plot": "boxplot"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"trend", "boxplot"}:
        raise PlotWorkspaceError(f"Unknown plot operation {value!r}.")
    return normalized


def _render_trend_png(request: TrendPlotRequest) -> bytes:
    """Render a canonical trend request through the core plot presentation."""

    cleaned = plotting._clean_trend_series(  # noqa: SLF001
        request.series,
        x_min=request.x_min,
        x_max=request.x_max,
    )
    if not cleaned:
        raise PlotWorkspaceError(
            f"No trend data found for metric {request.metric_name!r}."
        )
    axes = {item.axis for item in cleaned}
    if len(axes) != 1:
        raise PlotWorkspaceError(
            f"Trend metric {request.metric_name!r} uses mixed axes: "
            f"{', '.join(sorted(axes))}."
        )

    axis = next(iter(axes))
    fig, ax = plotting._new_figure()  # noqa: SLF001
    try:
        positive_values = True
        for index, item in enumerate(cleaned):
            values = item.mean
            positive_values = positive_values and bool(values) and min(values) > 0
            ax.plot(
                item.key,
                values,
                label=item.label,
                linewidth=2.0,
                linestyle=plotting._line_style(index),  # noqa: SLF001
                color=plotting._line_color(index),  # noqa: SLF001
            )
        if request.y_scale == "log" or (
            request.y_scale == "auto" and positive_values
        ):
            ax.set_yscale("log")
        ax.set_xlabel(plotting._axis_label(axis))  # noqa: SLF001
        ax.set_ylabel(plotting._metric_ylabel(request.metric_name))  # noqa: SLF001
        ax.set_title(
            request.title
            or f"Trend of {request.metric_name} Over {plotting._axis_label(axis)}"  # noqa: SLF001
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        plotting._apply_legend(  # noqa: SLF001
            ax,
            request.legend_position,
            len(cleaned),
        )
        fig.tight_layout()
        return _figure_png_bytes(fig)
    finally:
        _close_figure(fig)


def _render_boxplot_png(request: BoxplotRequest) -> bytes:
    """Render a canonical boxplot request through the core presentation."""

    labels = [item.label for item in request.series]
    values = [item.values for item in request.series]
    fig, ax = plotting._new_figure()  # noqa: SLF001
    try:
        ax.boxplot(values, notch=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(request.metric_name)
        ax.set_title(request.title or f"Boxplot of {request.metric_name}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)
        if len(labels) > 5:
            ax.tick_params(axis="x", rotation=45)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        return _figure_png_bytes(fig)
    finally:
        _close_figure(fig)


def _figure_png_bytes(fig: Any) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    content = buffer.getvalue()
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PlotWorkspaceError("Plot preview renderer produced an invalid PNG image.")
    return content


def _close_figure(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _require_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlotWorkspaceError("Plot payload must be an object.")
    return value


def _require_series_mapping(value: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlotWorkspaceError(f"series[{index}] must be an object.")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], *, prefix: str = ""
) -> None:
    unknown = sorted((str(key) for key in value if key not in allowed))
    if unknown:
        raise PlotWorkspaceError(
            f"{prefix}has unknown field(s): {', '.join(unknown)}."
        )


def _bounded_text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlotWorkspaceError(f"{name} must be a non-empty string.")
    text = value.strip()
    if "\x00" in text or len(text) > maximum:
        raise PlotWorkspaceError(f"{name} contains invalid or excessive text.")
    return text


def _optional_text(value: Any, name: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, name, maximum=maximum)


def _validate_formats(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("png",)
    if not isinstance(value, (list, tuple, set, frozenset)) or not value:
        raise PlotWorkspaceError("formats must be a non-empty array.")
    requested: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise PlotWorkspaceError("Every plot format must be a string.")
        normalized = item.strip().lower()
        if normalized not in PLOT_FORMATS:
            raise PlotWorkspaceError(f"Unknown plot format: {item!r}.")
        requested.add(normalized)
    return tuple(sorted(requested))


def _choice(value: Any, name: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise PlotWorkspaceError(f"{name} must be one of: {', '.join(sorted(choices))}.")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise PlotWorkspaceError(f"{name} must be one of: {', '.join(sorted(choices))}.")
    return normalized


def _optional_finite_number(value: Any, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise PlotWorkspaceError(f"{name} must be a finite number, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlotWorkspaceError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise PlotWorkspaceError(f"{name} must be a finite number.")
    return result


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise PlotWorkspaceError(f"{name} must be an array.")
    return value


def _key_list(value: Any, name: str) -> list[int | float | str]:
    values = _sequence(value, name)
    result: list[int | float | str] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float, str)):
            raise PlotWorkspaceError(
                f"{name}[{index}] must be a finite number or bounded string."
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise PlotWorkspaceError(f"{name}[{index}] must be finite.")
        if isinstance(item, str) and ("\x00" in item or len(item) > MAX_KEY_TEXT_LENGTH):
            raise PlotWorkspaceError(f"{name}[{index}] contains invalid or excessive text.")
        result.append(item)
    return result


def _is_finite_numeric_key(value: int | float | str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_number_list(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> list[float]:
    values = _sequence(value, name)
    result: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool):
            raise PlotWorkspaceError(f"{name}[{index}] must be a finite number.")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PlotWorkspaceError(f"{name}[{index}] must be a finite number.") from exc
        if not math.isfinite(number):
            raise PlotWorkspaceError(f"{name}[{index}] must be finite.")
        if minimum is not None and number < minimum:
            raise PlotWorkspaceError(f"{name}[{index}] must be at least {minimum:g}.")
        result.append(number)
    return result


def _optional_number_list(
    value: Any,
    name: str,
    *,
    expected_length: int,
    minimum: float | None = None,
) -> list[float] | None:
    if value is None:
        return None
    result = _finite_number_list(value, name, minimum=minimum)
    if len(result) != expected_length:
        raise PlotWorkspaceError(f"{name} must have {expected_length} items.")
    return result


def _optional_count_list(
    value: Any,
    name: str,
    *,
    expected_length: int,
) -> list[int] | None:
    if value is None:
        return None
    values = _sequence(value, name)
    result: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PlotWorkspaceError(
                f"{name}[{index}] must be a non-negative integer."
            )
        result.append(item)
    if len(result) != expected_length:
        raise PlotWorkspaceError(f"{name} must have {expected_length} items.")
    return result


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlotWorkspaceError(f"{name} must be a positive integer.")
    return value


def _validate_file_signature(path: Path, fmt: str) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(4_096)
    valid = (
        (fmt == "png" and prefix.startswith(b"\x89PNG\r\n\x1a\n"))
        or (fmt == "pdf" and prefix.startswith(b"%PDF-"))
        or (fmt == "svg" and b"<svg" in prefix.lower())
    )
    if not valid:
        raise PlotWorkspaceError(
            f"Plot exporter produced an invalid {fmt.upper()} artifact."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
