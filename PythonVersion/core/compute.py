"""Shared compute backend selection and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any

_DEFAULT_WORKER_THREADS = 1
_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
_THREAD_DYNAMIC_ENV_VARS = {
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
_THREADPOOL_LIMITER: Any | None = None


@dataclass(slots=True)
class ComputeConfig:
    mps_min_work_items: int = 10000


_CONFIG = ComputeConfig()
_TRACE: dict[str, dict[str, Any]] = {}


def reset_compute_trace() -> None:
    _TRACE.clear()


@lru_cache(maxsize=1)
def configure_worker_thread_limits() -> None:
    """Limit library thread pools for one experiment worker process.

    Batch execution parallelizes at the run level. Without a per-worker cap,
    BLAS/OpenMP/PyTorch may each create their own thread pool inside every
    process and oversubscribe the machine. The default keeps each run to one
    CPU thread; set ``EDOLAB_RUN_THREADS`` to a positive integer to opt into a
    wider per-run limit.
    """

    if _truthy_env("EDOLAB_DISABLE_WORKER_THREAD_LIMIT"):
        return

    worker_threads = _positive_int_env("EDOLAB_RUN_THREADS") or _DEFAULT_WORKER_THREADS
    _set_thread_limit_environment(worker_threads)
    _apply_threadpoolctl_limit(worker_threads)
    _configure_torch_threads(worker_threads)


@lru_cache(maxsize=1)
def configure_torch_threads_from_env() -> None:
    """Apply optional per-process torch thread limits from environment variables.

    Kept for compatibility with older predictor paths and tests. New run-level
    scheduling should call :func:`configure_worker_thread_limits`.
    """

    _configure_torch_threads(None)


def _set_thread_limit_environment(thread_count: int) -> None:
    value = str(max(1, int(thread_count)))
    for name in _THREAD_LIMIT_ENV_VARS:
        os.environ[name] = value
    for name, env_value in _THREAD_DYNAMIC_ENV_VARS.items():
        os.environ[name] = env_value


def _apply_threadpoolctl_limit(thread_count: int) -> None:
    global _THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return
    try:
        _THREADPOOL_LIMITER = threadpool_limits(limits=max(1, int(thread_count)))
    except Exception:
        _THREADPOOL_LIMITER = None


def _configure_torch_threads(default_threads: int | None) -> None:
    thread_count = _positive_int_env("EDOLAB_TORCH_THREADS")
    if thread_count is None and default_threads is not None:
        thread_count = max(1, int(default_threads))
    interop_count = _positive_int_env("EDOLAB_TORCH_INTEROP_THREADS")
    if interop_count is None and thread_count is not None:
        interop_count = thread_count
    if thread_count is None and interop_count is None:
        return
    torch, _error = _torch_result()
    if torch is None:
        return
    if thread_count is not None:
        try:
            torch.set_num_threads(thread_count)
        except RuntimeError:
            pass
    if interop_count is not None:
        try:
            torch.set_num_interop_threads(interop_count)
        except RuntimeError:
            pass


def record_compute_use(component: str, label: str, note: str | None = None) -> None:
    entry = _TRACE.setdefault(component, {"uses": {}, "notes": []})
    uses = entry["uses"]
    uses[label] = int(uses.get(label, 0)) + 1
    if note:
        record_compute_note(component, note)


def record_compute_note(component: str, note: str) -> None:
    entry = _TRACE.setdefault(component, {"uses": {}, "notes": []})
    notes = entry["notes"]
    if note not in notes and len(notes) < 8:
        notes.append(note)


def compute_report() -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name, entry in sorted(_TRACE.items()):
        components[name] = {
            "uses": dict(entry.get("uses", {})),
            "notes": list(entry.get("notes", [])),
        }
    return {
        "components": components,
        "summary": format_compute_report(
            {
                "components": components,
            }
        ),
    }


def format_compute_report(report: dict[str, Any] | None) -> str:
    if not report:
        return "compute=unknown"
    components = report.get("components", {})
    if not components:
        return "no accelerated components used"
    parts: list[str] = []
    for component, entry in sorted(components.items()):
        uses = entry.get("uses", {})
        if not uses:
            continue
        labels = sorted(uses, key=lambda label: (-int(uses[label]), label))
        parts.append(f"{component}=" + "|".join(labels))
    return ", ".join(parts)


def select_torch_device(
    component: str,
    *,
    work_items: int = 0,
    allow_torch_cpu: bool = True,
    device: str | None = None,
) -> tuple[Any | None, Any | None, str]:
    """Return a torch module/device pair or a CPU fallback decision.

    ``device`` is the component-level device policy.
    """

    requested_device = _component_device(device)
    if requested_device == "numpy":
        record_compute_use(component, "numpy:cpu")
        return None, None, "numpy:cpu"

    torch, error = _torch_result()
    if torch is None:
        record_compute_use(component, "numpy:cpu", "torch unavailable; using NumPy CPU")
        return None, None, "numpy:cpu"

    for candidate in _device_chain(requested_device):
        if candidate == "cuda" or candidate.startswith("cuda:"):
            if torch.cuda.is_available():
                device = torch.device(candidate)
                if device.index is not None and device.index >= torch.cuda.device_count():
                    record_compute_note(component, f"{candidate} unavailable")
                    continue
                label = f"torch:{candidate}"
                record_compute_use(component, label)
                return torch, device, label
            record_compute_note(component, f"{candidate} unavailable")
        elif candidate == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            use_mps = has_mps and (
                requested_device != "auto" or int(work_items) >= _CONFIG.mps_min_work_items
            )
            if use_mps:
                device = torch.device("mps")
                record_compute_use(component, "torch:mps")
                return torch, device, "torch:mps"
            if has_mps:
                record_compute_note(component, "mps available but batch is too small")
            else:
                record_compute_note(component, "mps unavailable")
        elif candidate == "cpu":
            if allow_torch_cpu:
                device = torch.device("cpu")
                record_compute_use(component, "torch:cpu")
                return torch, device, "torch:cpu"
            record_compute_use(component, "numpy:cpu")
            return None, None, "numpy:cpu"

    note = f"torch import failed: {error}" if error is not None else "no usable torch device"
    record_compute_use(component, "numpy:cpu", note)
    return None, None, "numpy:cpu"


def set_mps_min_work_items(value: int) -> None:
    _CONFIG.mps_min_work_items = max(0, int(value))


def normalize_torch_device_policy(
    value: str | None,
    *,
    allow_numpy: bool = True,
) -> str:
    """Normalize a user-facing torch device policy."""

    lowered = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "gpu": "cuda",
        "nvidia": "cuda",
        "msp": "mps",
        "metal": "mps",
        "apple": "mps",
    }
    resolved = aliases.get(lowered, lowered)
    if resolved.startswith("gpu:") and resolved[4:].isdigit():
        resolved = f"cuda:{resolved[4:]}"
    elif resolved.startswith("gpu") and resolved[3:].isdigit():
        resolved = f"cuda:{resolved[3:]}"
    if resolved.startswith("cuda:"):
        index = resolved.split(":", 1)[1]
        if index.isdigit():
            return resolved
        raise ValueError("cuda device policy must use cuda:<non-negative-id>")
    allowed = {"auto", "cuda", "mps", "cpu"}
    if allow_numpy:
        allowed.add("numpy")
    if resolved not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(f"device policy must be one of: {values}, or cuda:<id>")
    return resolved


def _normalize_device(value: str) -> str:
    try:
        return normalize_torch_device_policy(value, allow_numpy=True)
    except ValueError:
        return "auto"


def _component_device(value: str | None) -> str:
    if value is None:
        return "auto"
    lowered = str(value).strip().lower()
    if lowered in {"", "default"}:
        return "auto"
    return _normalize_device(lowered)


def _device_chain(device: str) -> list[str]:
    if device.startswith("cuda:"):
        return [device, "cpu"]
    if device == "cuda":
        return ["cuda", "mps", "cpu"]
    if device == "mps":
        return ["mps", "cpu"]
    if device == "cpu":
        return ["cpu"]
    if device == "numpy":
        return ["numpy"]
    return ["cuda", "mps", "cpu"]


def _positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        number = int(str(value).strip())
    except ValueError:
        return None
    if number < 1:
        return None
    return number


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _torch_result() -> tuple[Any | None, Exception | None]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional package.
        return None, exc
    return torch, None
