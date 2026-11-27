"""FastAPI adapter for the EDOLAB Relay application service."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import asynccontextmanager
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import quote, unquote

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import RelayPaths
from .performance import metrics, sample_stacks
from .service import (
    Controller,
    RESULT_ARCHIVE_MAX_UPLOAD_BYTES,
    RelayService,
    ServiceError,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerInput(_StrictModel):
    name: str = Field(min_length=1, max_length=200)
    host: str = Field(min_length=1, max_length=253)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=128)
    ssh_key_path: str = Field(min_length=1, max_length=4096)
    remote_root: str = Field(min_length=1, max_length=4096)
    enabled: bool = True
    conda_env: str | None = Field(default=None, min_length=1, max_length=128)
    poll_interval_seconds: float | None = Field(default=None, gt=0, le=300)
    default_max_parallel_runs: int | None = Field(default=None, ge=1, le=256)
    min_free_bytes: int | None = Field(default=None, ge=0)
    max_load_ratio: float | None = Field(default=None, gt=0)


class CreateRunInput(_StrictModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    task_path: str | None = Field(default=None, min_length=1, max_length=4096)
    task_jsonl: str | None = Field(
        default=None,
        min_length=1,
        max_length=2 * 1024 * 1024,
    )
    server_id: str = Field(min_length=1, max_length=200)
    max_parallel_runs: int = Field(default=1, ge=1, le=256)
    refresh_interval_seconds: float = Field(default=5, ge=1, le=300)
    admission_override: bool = False

    @model_validator(mode="after")
    def require_exactly_one_task_source(self):
        has_path = bool(self.task_path and self.task_path.strip())
        has_jsonl = bool(self.task_jsonl and self.task_jsonl.strip())
        if has_path == has_jsonl:
            raise ValueError("exactly one of task_path or task_jsonl is required")
        return self


class RunDisplayNameInput(_StrictModel):
    display_name: str = Field(min_length=1, max_length=200)


class BenchmarkInput(_StrictModel):
    problem_name: str = Field(min_length=1, max_length=200)
    problem_params: dict[str, Any] = Field(default_factory=dict)
    environment_number: int = Field(default=1, ge=1, le=1_000_000)
    seed: int | None = Field(default=None, ge=0, le=(1 << 63) - 1)
    resolution: int = Field(default=80, ge=1, le=10_000)
    sample_count: int = Field(default=800, ge=1, le=1_000_000)


class PlotInput(_StrictModel):
    kind: str = Field(min_length=1, max_length=32)
    metric_name: str = Field(min_length=1, max_length=256)
    series: list[dict[str, Any]] = Field(min_length=1, max_length=80)
    formats: list[str] = Field(min_length=1, max_length=3)
    title: str | None = Field(default=None, max_length=2_048)
    x_min: float | None = None
    x_max: float | None = None
    legend_position: str | None = Field(default=None, min_length=1, max_length=64)
    y_scale: str | None = Field(default=None, min_length=1, max_length=32)


class PlotPreviewInput(_StrictModel):
    kind: str = Field(min_length=1, max_length=32)
    metric_name: str = Field(min_length=1, max_length=256)
    series: list[dict[str, Any]] = Field(min_length=1, max_length=80)
    title: str | None = Field(default=None, max_length=2_048)
    x_min: float | None = None
    x_max: float | None = None
    legend_position: str | None = Field(default=None, min_length=1, max_length=64)
    y_scale: str | None = Field(default=None, min_length=1, max_length=32)


class ResultImportInput(_StrictModel):
    source_path: str = Field(min_length=1, max_length=4096)
    display_name: str | None = Field(default=None, min_length=1, max_length=200)


class ResultImportFromRunInput(_StrictModel):
    run_id: str = Field(min_length=1, max_length=200)
    display_name: str | None = Field(default=None, min_length=1, max_length=200)


class ResultImportFromCandidateInput(_StrictModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    display_name: str | None = Field(default=None, min_length=1, max_length=200)


class ResultRunsetExportSelection(_StrictModel):
    result_id: str = Field(min_length=1, max_length=200)
    runset_uids: list[str] = Field(min_length=1, max_length=10_000)


class ResultExportInput(_StrictModel):
    result_ids: list[str] | None = Field(default=None, min_length=1, max_length=100)
    selections: list[ResultRunsetExportSelection] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def require_one_export_scope(self):
        if bool(self.result_ids) == bool(self.selections):
            raise ValueError("exactly one of result_ids or selections is required")
        return self


class TaskCreateInput(_StrictModel):
    filename: str = Field(min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=2 * 1024 * 1024)


class TaskUpdateInput(_StrictModel):
    filename: str | None = Field(default=None, min_length=1, max_length=240)
    content: str = Field(min_length=1, max_length=2 * 1024 * 1024)


class SAImportSelection(_StrictModel):
    result_id: str = Field(min_length=1, max_length=200)
    runset_uids: list[str] = Field(min_length=1, max_length=10_000)


class SAImportInput(_StrictModel):
    result_id: str | None = Field(default=None, min_length=1, max_length=200)
    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    runset_uids: list[str] | None = None
    selections: list[SAImportSelection] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )
    allow_duplicates: bool = False

    @model_validator(mode="after")
    def require_result_identity(self):
        has_legacy_identity = bool(self.result_id or self.run_id)
        if has_legacy_identity == bool(self.selections):
            raise ValueError("exactly one of result_id or selections is required")
        if self.result_id and self.run_id and self.result_id != self.run_id:
            raise ValueError("result_id and legacy run_id must match")
        return self

    @property
    def canonical_result_id(self) -> str:
        return str(self.result_id or self.run_id)


class SAMatchInput(_StrictModel):
    base_import_id: str | None = Field(default=None, min_length=1, max_length=200)
    method: str = Field(default="friedman", min_length=1, max_length=64)


class SAQuickFilterInput(_StrictModel):
    allow_duplicates: StrictBool = False


class SAAnalysisInput(_StrictModel):
    name: str = Field(min_length=1, max_length=200)
    method: str = Field(default="friedman", min_length=1, max_length=64)
    import_ids: list[str] = Field(min_length=2)
    metrics: list[str] | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)
    formats: list[str] | None = None
    label_parameter: str | None = Field(default=None, min_length=1, max_length=256)


class SettingsInput(_StrictModel):
    workspace_path: str | None = None
    default_ssh_key_path: str = Field(min_length=1, max_length=4096)
    default_remote_root: str = Field(min_length=1, max_length=4096)
    default_poll_interval_seconds: float = Field(ge=1, le=300)
    resource_sampling_interval_seconds: float = Field(default=3, ge=1, le=300)
    resource_history_retention_hours: float = Field(default=72, ge=1, le=720)
    default_max_parallel_runs: int = Field(ge=1, le=256)
    probe_timeout_seconds: float = Field(ge=3, le=300)
    require_clean_git: bool
    verify_artifact_checksums: bool
    delete_remote_results_after_transfer: bool


def create_app(
    *,
    service: RelayService | None = None,
    paths: RelayPaths | None = None,
    controller: Controller | None = None,
    trusted_hosts: Sequence[str] = ("127.0.0.1", "localhost", "[::1]", "testserver"),
) -> FastAPI:
    relay_paths = paths or RelayPaths.resolve()
    relay_service = service or RelayService(relay_paths, controller=controller)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            relay_service.start()
            yield
        finally:
            relay_service.stop()

    app = FastAPI(
        title="EDOLAB Relay",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.relay_service = relay_service
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))

    @app.middleware("http")
    async def measure_requests(request: Request, call_next):
        start = time.perf_counter()
        metrics.request_started()
        failed = True
        try:
            response = await call_next(request)
            failed = response.status_code >= 500
            return response
        finally:
            route = request.scope.get("route")
            name = getattr(route, "path", "unmatched")
            metrics.record(f"http.{request.method} {name}", time.perf_counter() - start, failed=failed)
            metrics.request_finished()

    @app.get("/api/performance")
    def performance() -> dict[str, Any]:
        return metrics.snapshot()

    @app.get("/api/performance/stacks")
    def performance_stacks(seconds: float = Query(5.0, ge=0.5, le=30.0)) -> dict[str, Any]:
        """Where every Python thread spends its time: a short sampling profile of the live process."""
        return sample_stacks(seconds)

    @app.middleware("http")
    async def reject_cross_site_mutations(request: Request, call_next):
        """Keep untrusted web pages from driving the localhost controller.

        Relay intentionally has no login because it only binds to loopback.
        Loopback alone does not stop a different web origin from submitting an
        HTML form to a body-less POST endpoint such as probe or cancel.  Modern
        browsers provide both ``Origin`` and ``Sec-Fetch-Site`` for those
        requests, so reject cross-origin mutations before they reach a route.
        Non-browser localhost clients, which commonly omit both headers, remain
        supported.
        """

        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
            origin = request.headers.get("origin", "").strip().rstrip("/")
            host = request.headers.get("host", "").strip()
            expected_origin = (
                f"{request.url.scheme}://{host}".rstrip("/") if host else ""
            )
            if fetch_site in {"cross-site", "same-site"} or (
                origin and origin != expected_origin
            ):
                response = JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Cross-origin browser mutations are not allowed.",
                        "code": "cross_origin_mutation_forbidden",
                    },
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(_, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
        details = []
        for item in exc.errors():
            location = ".".join(str(part) for part in item.get("loc", ()) if part != "body")
            message = str(item.get("msg", "invalid value"))
            details.append(f"{location}: {message}" if location else message)
        return JSONResponse(
            status_code=422,
            content={"detail": "; ".join(details), "code": "validation_error"},
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            schema_version = relay_service.database.schema_version
            if callable(schema_version):
                schema_version = schema_version()
            if int(schema_version) < 1:
                raise RuntimeError("remote database schema is not initialized")
            database_status = "ok"
            status = "ok"
        except Exception:
            database_status = "error"
            status = "degraded"
        return {
            "status": status,
            "service": "EDOLAB Relay",
            "version": app.version,
            "database": database_status,
            "workspace": str(relay_service.paths.repository_root),
            "demo": False,
        }

    @app.get("/api/dashboard")
    def dashboard() -> dict[str, Any]:
        return relay_service.dashboard()

    @app.get("/api/servers")
    def list_servers() -> list[dict[str, Any]]:
        return relay_service.list_servers()

    @app.get("/api/servers/resources/latest")
    def list_server_resource_monitors() -> dict[str, Any]:
        return relay_service.list_server_resource_monitors()

    @app.post("/api/servers", status_code=201)
    def create_server(payload: ServerInput) -> dict[str, Any]:
        return relay_service.create_server(payload.model_dump(exclude_none=True))

    @app.get("/api/servers/{server_id}")
    def get_server(server_id: str) -> dict[str, Any]:
        return relay_service.get_server(server_id)

    @app.put("/api/servers/{server_id}")
    def update_server(server_id: str, payload: ServerInput) -> dict[str, Any]:
        return relay_service.update_server(
            server_id, payload.model_dump(exclude_none=True)
        )

    @app.delete("/api/servers/{server_id}", status_code=204)
    def delete_server(server_id: str) -> Response:
        relay_service.delete_server(server_id)
        return Response(status_code=204)

    @app.post("/api/servers/{server_id}/probe")
    def probe_server(server_id: str) -> dict[str, Any]:
        return relay_service.probe_server(server_id)

    @app.post("/api/servers/{server_id}/resources/refresh")
    def refresh_server_resources(
        server_id: str,
        request: Request,
    ) -> dict[str, Any]:
        if request.headers.get("x-edolab-manual-sample") != "1":
            raise ServiceError(
                400,
                "manual_sample_confirmation_required",
                "Immediate SSH sampling requires the explicit "
                "X-EDOLAB-Manual-Sample: 1 header. Dashboard polling must use "
                "the read-only database endpoints.",
            )
        return relay_service.refresh_server_resources(server_id)

    @app.get("/api/servers/{server_id}/resource-samples")
    def get_server_resource_history(
        server_id: str,
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
        after_id: int | None = Query(default=None, ge=0),
        limit: int = Query(default=5_000, ge=1, le=10_000),
    ) -> dict[str, Any]:
        return relay_service.get_server_resource_history(
            server_id,
            since=since,
            until=until,
            after_id=after_id,
            limit=limit,
        )

    @app.get("/api/tasks")
    def list_tasks() -> list[dict[str, Any]]:
        return relay_service.list_tasks()

    @app.post("/api/tasks/inspect")
    def inspect_task(payload: CreateRunInput) -> dict[str, Any]:
        return relay_service.inspect_task(payload.model_dump())

    @app.post("/api/task-files", status_code=201)
    def create_task_document(payload: TaskCreateInput) -> dict[str, Any]:
        return relay_service.create_task_document(payload.model_dump())

    @app.get("/api/task-files/{filename}")
    def get_task_document(filename: str) -> dict[str, Any]:
        return relay_service.get_task_document(f"tasks/{filename}")

    @app.put("/api/task-files/{filename}")
    def update_task_document(
        filename: str,
        payload: TaskUpdateInput,
    ) -> dict[str, Any]:
        return relay_service.update_task_document(
            f"tasks/{filename}",
            payload.model_dump(exclude_none=True),
        )

    @app.delete("/api/task-files/{filename}")
    def delete_task_document(filename: str) -> dict[str, Any]:
        return relay_service.delete_task_document(f"tasks/{filename}")

    @app.get("/api/runs")
    def list_runs(
        limit: int = Query(default=500, ge=1, le=1_000),
        offset: int = Query(default=0, ge=0),
        status: str = Query(default="all"),
        search: str | None = Query(default=None, max_length=200),
    ) -> list[dict[str, Any]]:
        return relay_service.list_runs(limit=limit, offset=offset, status=status, search=search)

    @app.post("/api/runs", status_code=201)
    def create_run(payload: CreateRunInput) -> dict[str, Any]:
        return relay_service.create_run(payload.model_dump())

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return relay_service.get_run(run_id)

    @app.put("/api/runs/{run_id}/display-name")
    def rename_run(run_id: str, payload: RunDisplayNameInput) -> dict[str, Any]:
        return relay_service.rename_run(run_id, payload.display_name)

    @app.get("/api/runs/{run_id}/events")
    def list_run_events(
        run_id: str,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=10_000, ge=1, le=10_000),
    ) -> list[dict[str, Any]]:
        return relay_service.list_run_events(
            run_id, after_sequence=after_sequence, limit=limit
        )

    @app.get("/api/runs/{run_id}/samples")
    def list_run_samples(
        run_id: str,
        after_id: int | None = Query(default=None, ge=0),
        limit: int = Query(default=10_000, ge=1, le=10_000),
    ) -> list[dict[str, Any]]:
        return relay_service.list_run_samples(run_id, after_id=after_id, limit=limit)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        return relay_service.cancel_run(run_id)

    @app.post("/api/runs/{run_id}/recover")
    def recover_run(run_id: str) -> dict[str, Any]:
        return relay_service.recover_run(run_id)

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> Response:
        relay_service.delete_run(run_id)
        return Response(status_code=204)

    @app.get("/api/plots/sources")
    def list_plot_sources() -> list[dict[str, Any]]:
        return relay_service.list_plot_sources()

    @app.get("/api/plots/series/trend")
    def load_plot_trend_series(
        source_id: str = Query(min_length=1, max_length=4_096),
        metric_name: str = Query(min_length=1, max_length=256),
        x_min: float | None = Query(default=None),
        x_max: float | None = Query(default=None),
        max_points: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        return relay_service.load_plot_trend_series(
            source_id,
            metric_name,
            x_min=x_min,
            x_max=x_max,
            max_points=max_points,
        )

    @app.get("/api/plots/series/boxplot")
    def load_plot_boxplot_series(
        source_id: str = Query(min_length=1, max_length=4_096),
        metric_name: str = Query(min_length=1, max_length=256),
    ) -> dict[str, Any]:
        return relay_service.load_plot_boxplot_series(source_id, metric_name)

    @app.post("/api/plots")
    def render_plot(payload: PlotInput) -> dict[str, Any]:
        return relay_service.render_plot(payload.model_dump(exclude_none=True))

    @app.post("/api/plots/preview", response_class=Response)
    def preview_plot(payload: PlotPreviewInput) -> Response:
        preview = relay_service.render_plot_preview(
            payload.model_dump(exclude_none=True)
        )
        headers = {
            "Cache-Control": "no-store",
            "X-EDOLAB-Plot-Kind": preview.kind,
            "X-EDOLAB-Metric-Name": quote(preview.metric_name, safe="._-"),
            "X-EDOLAB-Series-Count": str(preview.series_count),
            "X-EDOLAB-Point-Count": str(preview.point_count),
        }
        if preview.axis is not None:
            headers["X-EDOLAB-Axis"] = quote(preview.axis, safe="._-")
        return Response(
            content=preview.content,
            media_type="image/png",
            headers=headers,
        )

    @app.get("/api/plots/artifacts/{storage_key:path}")
    def get_interactive_plot_artifact(storage_key: str) -> FileResponse:
        path, media_type = relay_service.resolve_interactive_plot_artifact(
            storage_key
        )
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Cache-Control": "private, max-age=3600",
                "Content-Disposition": f'inline; filename="{path.name}"',
            },
        )

    @app.get("/api/plots/runs/{run_id}")
    def get_plot_run_result(run_id: str) -> dict[str, Any]:
        return relay_service.get_plot_run_result(run_id)

    @app.get("/api/plots/runs/{run_id}/artifacts/{storage_key:path}")
    def get_plot_run_artifact(run_id: str, storage_key: str) -> FileResponse:
        path, media_type = relay_service.resolve_plot_run_artifact(run_id, storage_key)
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Cache-Control": "private, max-age=31536000, immutable",
                "Content-Disposition": f'inline; filename="{path.name}"',
            },
        )

    @app.get("/api/benchmarks")
    def benchmark_catalog() -> dict[str, Any]:
        return relay_service.benchmark_catalog()

    @app.post("/api/benchmarks/{operation}")
    def render_benchmark(
        operation: str,
        payload: BenchmarkInput,
    ) -> dict[str, Any]:
        return relay_service.render_benchmark(
            operation,
            payload.model_dump(exclude_none=True),
        )

    @app.get("/api/benchmarks/artifacts/{storage_key:path}")
    def get_interactive_benchmark_artifact(storage_key: str) -> FileResponse:
        path, media_type = relay_service.resolve_interactive_benchmark_artifact(
            storage_key
        )
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Cache-Control": "private, max-age=3600",
                "Content-Disposition": f'inline; filename="{path.name}"',
            },
        )

    @app.get("/api/benchmarks/runs/{run_id}")
    def get_benchmark_run_result(run_id: str) -> dict[str, Any]:
        return relay_service.get_benchmark_run_result(run_id)

    @app.get("/api/benchmarks/runs/{run_id}/artifacts/{storage_key:path}")
    def get_benchmark_run_artifact(run_id: str, storage_key: str) -> FileResponse:
        path, media_type, encoding = relay_service.resolve_benchmark_run_artifact(
            run_id,
            storage_key,
        )
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "Content-Disposition": f'inline; filename="{path.name}"',
        }
        if encoding:
            headers["Content-Encoding"] = encoding
        return FileResponse(path, media_type=media_type, headers=headers)

    @app.get("/api/results/catalog")
    def list_result_catalog() -> list[dict[str, Any]]:
        return relay_service.list_result_catalog()

    @app.get("/api/results/candidates")
    def list_result_candidates() -> list[dict[str, Any]]:
        return relay_service.list_result_candidates()

    @app.post("/api/results/imports", status_code=201)
    def import_result_bundle(payload: ResultImportInput) -> dict[str, Any]:
        return relay_service.import_result_bundle(
            payload.model_dump(exclude_none=True)
        )

    @app.post("/api/results/imports/upload")
    async def upload_result_archive(request: Request) -> dict[str, Any]:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type not in {
            "application/zip",
            "application/x-zip-compressed",
            "application/octet-stream",
        }:
            raise ServiceError(
                415,
                "result_upload_requires_zip",
                "External result upload must be a ZIP file.",
            )
        encoded_filename = request.headers.get("x-edolab-filename", "")
        try:
            filename = unquote(encoded_filename)
        except Exception:
            filename = encoded_filename
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise ServiceError(
                    400,
                    "invalid_result_upload_size",
                    "Uploaded ZIP has an invalid Content-Length.",
                ) from exc
            if declared_size <= 0 or declared_size > RESULT_ARCHIVE_MAX_UPLOAD_BYTES:
                raise ServiceError(
                    413,
                    "result_upload_too_large",
                    "Uploaded ZIP is empty or exceeds the supported size.",
                )

        descriptor, raw_path = tempfile.mkstemp(
            prefix="result-upload-",
            suffix=".zip",
            dir=relay_service.paths.working,
        )
        upload_path = Path(raw_path)
        upload_path.chmod(0o600)
        received = 0
        try:
            async for chunk in request.stream():
                if not chunk:
                    continue
                received += len(chunk)
                if received > RESULT_ARCHIVE_MAX_UPLOAD_BYTES:
                    raise ServiceError(
                        413,
                        "result_upload_too_large",
                        "Uploaded ZIP exceeds the supported size.",
                    )
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
            if received == 0:
                raise ServiceError(
                    400,
                    "result_upload_empty",
                    "Uploaded ZIP is empty.",
                )
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            return relay_service.import_uploaded_result_archive(
                upload_path,
                original_filename=filename,
            )
        except OSError as exc:
            raise ServiceError(
                409,
                "result_upload_write_failed",
                f"Uploaded ZIP could not be staged safely: {exc}",
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            upload_path.unlink(missing_ok=True)

    @app.post("/api/results/imports/from-run", status_code=201)
    def import_result_from_run(payload: ResultImportFromRunInput) -> dict[str, Any]:
        return relay_service.import_result_from_run(
            payload.model_dump(exclude_none=True)
        )

    @app.post("/api/results/imports/from-candidate", status_code=201)
    def import_result_from_candidate(
        payload: ResultImportFromCandidateInput,
    ) -> dict[str, Any]:
        return relay_service.import_result_candidate(
            payload.model_dump(exclude_none=True)
        )

    @app.get("/api/results/{result_id}")
    def preview_imported_result(result_id: str) -> dict[str, Any]:
        return relay_service.preview_run_results(result_id)

    @app.get("/api/results/{result_id}/runsets/{runset_uid}/samples")
    def preview_imported_result_samples(
        result_id: str,
        runset_uid: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1_000),
    ) -> dict[str, Any]:
        return relay_service.preview_runset_samples(
            result_id, runset_uid, offset=offset, limit=limit
        )

    @app.get("/api/results/{result_id}/runsets/{runset_uid}/trends/{metric_name}")
    def preview_imported_result_trend(
        result_id: str,
        runset_uid: str,
        metric_name: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=1_000, ge=1, le=10_000),
    ) -> dict[str, Any]:
        return relay_service.preview_runset_trend(
            result_id,
            runset_uid,
            metric_name,
            offset=offset,
            limit=limit,
        )

    @app.delete("/api/results/{result_id}", status_code=204)
    def delete_imported_result(result_id: str) -> Response:
        relay_service.delete_result(result_id)
        return Response(status_code=204)

    @app.post("/api/results/exports", response_class=FileResponse)
    def export_imported_results(payload: ResultExportInput) -> FileResponse:
        if payload.selections:
            archive_path, filename = relay_service.export_runsets(
                [item.model_dump() for item in payload.selections]
            )
        else:
            archive_path, filename = relay_service.export_results(payload.result_ids or [])
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=filename,
            headers={
                "Cache-Control": "no-store",
            },
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    @app.get("/api/sa/imports")
    def list_sa_imports() -> list[dict[str, Any]]:
        return relay_service.list_sa_imports()

    @app.post("/api/sa/imports", status_code=201)
    def import_sa_runsets(payload: SAImportInput) -> dict[str, Any]:
        if payload.selections:
            return relay_service.import_sa_runset_selections(
                [item.model_dump() for item in payload.selections],
                allow_duplicates=payload.allow_duplicates,
            )
        return relay_service.import_sa_runsets(
            payload.canonical_result_id,
            payload.runset_uids,
            allow_duplicates=payload.allow_duplicates,
        )

    @app.delete("/api/sa/imports/{import_id}", status_code=204)
    def delete_sa_import(import_id: str) -> Response:
        relay_service.delete_sa_import(import_id)
        return Response(status_code=204)

    @app.delete("/api/sa/imports", status_code=204)
    def clear_sa_imports() -> Response:
        relay_service.clear_sa_imports()
        return Response(status_code=204)

    @app.post("/api/sa/match")
    def match_sa_imports(payload: SAMatchInput) -> dict[str, Any]:
        return relay_service.match_sa_imports(
            base_import_id=payload.base_import_id,
            method=payload.method,
        )

    @app.post("/api/sa/quick-filter")
    def quick_filter_sa_imports(payload: SAQuickFilterInput) -> dict[str, Any]:
        return relay_service.quick_filter_sa_imports(
            allow_duplicates=payload.allow_duplicates
        )

    @app.get("/api/sa/analyses")
    def list_sa_analyses() -> list[dict[str, Any]]:
        return relay_service.list_sa_analyses()

    @app.post("/api/sa/analyses", status_code=201)
    def create_sa_analysis(payload: SAAnalysisInput) -> dict[str, Any]:
        return relay_service.create_sa_analysis(payload.model_dump())

    @app.get("/api/sa/analyses/{analysis_id}")
    def get_sa_analysis(analysis_id: str) -> dict[str, Any]:
        return relay_service.get_sa_analysis(analysis_id)

    @app.delete("/api/sa/analyses/{analysis_id}", status_code=204)
    def delete_sa_analysis(analysis_id: str) -> Response:
        relay_service.delete_sa_analysis(analysis_id)
        return Response(status_code=204)

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return relay_service.get_settings()

    @app.put("/api/settings")
    def save_settings(payload: SettingsInput) -> dict[str, Any]:
        return relay_service.save_settings(payload.model_dump())

    _install_frontend(app, relay_service.paths.frontend_dist)
    return app


def _install_frontend(app: FastAPI, dist: Path) -> None:
    assets = dist / "assets"
    index = dist / "index.html"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="relay-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(
            """
            <!doctype html><html><head><meta charset="utf-8"><title>EDOLAB Relay</title></head>
            <body style="font-family:system-ui;max-width:720px;margin:12vh auto;padding:24px">
              <h1>EDOLAB Relay API is running</h1>
              <p>The frontend build was not found. Build <code>apps/remote/frontend</code>
              or open <a href="/api/docs">/api/docs</a>.</p>
            </body></html>
            """,
            status_code=200,
        )


__all__ = ["CreateRunInput", "ServerInput", "SettingsInput", "create_app"]
