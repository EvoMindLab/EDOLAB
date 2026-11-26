"""Local-first remote execution subsystem for EDOLAB.

The persistence and domain layers are usable from either the Relay web server
or a one-shot Python/CLI entry point. Retired application services are not part
of this package.
"""

from .database import RemoteDatabase
from .domain import (
    ALLOWED_RUN_TRANSITIONS,
    NON_TERMINAL_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    Artifact,
    ArtifactStatus,
    DashboardCounts,
    EventLevel,
    ResourceSample,
    Run,
    RunEvent,
    RunStatus,
    Server,
    stable_json_dumps,
    utc_now,
    validate_run_transition,
)

__all__ = [
    "ALLOWED_RUN_TRANSITIONS",
    "NON_TERMINAL_RUN_STATUSES",
    "TERMINAL_RUN_STATUSES",
    "Artifact",
    "ArtifactStatus",
    "DashboardCounts",
    "EventLevel",
    "RemoteDatabase",
    "ResourceSample",
    "Run",
    "RunEvent",
    "RunStatus",
    "Server",
    "stable_json_dumps",
    "utc_now",
    "validate_run_transition",
]
