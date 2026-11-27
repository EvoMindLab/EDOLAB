"""Command-line launcher for the EDOLAB Relay local web application."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from typing import Sequence
import webbrowser

import uvicorn

from .api import create_app
from .config import ControllerAlreadyRunning, ControllerLock, RelayPaths
from .database import RemoteDatabase
from .service import Controller, RelayService


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m apps.remote",
        description="EDOLAB Relay local SSH experiment controller",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="start the local Relay web application")
    start.add_argument("--host", default="127.0.0.1", choices=sorted(LOOPBACK_HOSTS))
    start.add_argument("--port", type=int, default=8766)
    start.add_argument("--data-root", default=None)
    start.add_argument("--open-browser", action="store_true")
    start.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def run_relay(
    *,
    paths: RelayPaths,
    host: str = "127.0.0.1",
    port: int = 8766,
    open_browser: bool = False,
    log_level: str = "info",
    controller: Controller | None = None,
) -> None:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("Relay has no remote authentication and must bind to localhost.")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    paths.ensure()
    database = RemoteDatabase(paths.database)
    if controller is None:
        from .orchestrator import RelayController

        controller = RelayController(database, paths)
    service = RelayService(
        paths,
        database=database,
        controller=controller,
        enable_local_target=True,
    )
    trusted = tuple({"127.0.0.1", "localhost", "[::1]", host})
    app = create_app(service=service, trusted_hosts=trusted)
    lock = ControllerLock(paths.data_root / "control.lock")
    with lock:
        if open_browser:
            thread = threading.Thread(
                target=_open_browser_when_ready,
                args=(host, port),
                name="relay-browser-opener",
                daemon=True,
            )
            thread.start()
        uvicorn.run(app, host=host, port=port, log_level=log_level)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "start":
        parser.error(f"unsupported command: {args.command}")
    paths = RelayPaths.resolve(data_root=args.data_root)
    try:
        run_relay(
            paths=paths,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
            log_level=args.log_level,
        )
    except (ControllerAlreadyRunning, ValueError) as exc:
        print(f"EDOLAB Relay: {exc}", file=sys.stderr)
        return 2
    return 0


def _open_browser_when_ready(host: str, port: int) -> None:
    connect_host = "::1" if host == "::1" else "127.0.0.1"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.1)
    display_host = f"[{host}]" if ":" in host else host
    webbrowser.open(f"http://{display_host}:{port}/")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_relay"]
