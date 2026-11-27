"""Optional versioned per-run event records within the standard summary DB.

The SQLite tables are canonical so two-database bundle transfers preserve all
events. ``run_events.jsonl`` is a lossless, regenerable sidecar, never required
to validate an old bundle. Search-specific schemas belong to each algorithm.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3


def prepare_event_streams(streams):
    prepared = []
    if not isinstance(streams, dict):
        raise ValueError('event_streams must be a mapping')
    for name, stream in streams.items():
        if not isinstance(name, str) or not name or not isinstance(stream, dict):
            raise ValueError('Invalid event stream name or definition')
        version, events = stream.get('schema_version'), stream.get('events')
        if type(version) is not int or version < 1 or not isinstance(events, list):
            raise ValueError('Event streams require a positive schema_version and events list')
        if not all(isinstance(event, dict) for event in events):
            raise ValueError('Each event must be an object')
        payloads = [json.dumps(event, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False) for event in events]
        prepared.append((name, version, payloads))
    return prepared


def write_event_streams(connection, run_uid, prepared):
    if not prepared:
        return
    # Individual statements keep the caller's run-result transaction intact;
    # executescript would implicitly commit it before writing the event rows.
    connection.execute('''CREATE TABLE IF NOT EXISTS run_event_streams (
        run_uid TEXT NOT NULL REFERENCES runs(run_uid),
        stream TEXT NOT NULL, schema_version INTEGER NOT NULL,
        event_count INTEGER NOT NULL CHECK(event_count >= 0),
        PRIMARY KEY(run_uid, stream))''')
    connection.execute('''CREATE TABLE IF NOT EXISTS run_event_records (
        run_uid TEXT NOT NULL, stream TEXT NOT NULL,
        event_index INTEGER NOT NULL CHECK(event_index >= 1),
        payload_json TEXT NOT NULL,
        PRIMARY KEY(run_uid, stream, event_index),
        FOREIGN KEY(run_uid, stream) REFERENCES run_event_streams(run_uid, stream))''')
    for name, version, payloads in prepared:
        connection.execute('INSERT INTO run_event_streams VALUES (?, ?, ?, ?)',
                           (run_uid, name, version, len(payloads)))
        connection.executemany('INSERT INTO run_event_records VALUES (?, ?, ?, ?)',
            ((run_uid, name, index, payload) for index, payload in enumerate(payloads, 1)))


def has_event_streams(connection):
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_event_streams'"
    ).fetchone() is not None


def validate_event_streams(connection):
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'run_event_%'")}
    if not tables:
        return
    if not {'run_event_streams', 'run_event_records'}.issubset(tables):
        raise ValueError('Incomplete run event schema')
    invalid = connection.execute('''SELECT COUNT(*) FROM run_event_streams AS s
        LEFT JOIN runs AS r ON r.run_uid=s.run_uid
        WHERE r.run_uid IS NULL OR s.schema_version < 1 OR s.event_count < 0 OR
        s.event_count != (SELECT COUNT(*) FROM run_event_records AS e
                         WHERE e.run_uid=s.run_uid AND e.stream=s.stream) OR
        s.event_count != COALESCE((SELECT MAX(event_index) FROM run_event_records AS e
                         WHERE e.run_uid=s.run_uid AND e.stream=s.stream), 0)''').fetchone()[0]
    orphaned = connection.execute('''SELECT COUNT(*) FROM run_event_records AS e
        LEFT JOIN run_event_streams AS s ON s.run_uid=e.run_uid AND s.stream=e.stream
        WHERE s.run_uid IS NULL OR e.event_index < 1''').fetchone()[0]
    if invalid or orphaned:
        raise ValueError('Run event streams contain orphaned, missing or misindexed records')
    for (payload,) in connection.execute('SELECT payload_json FROM run_event_records'):
        def invalid_constant(value):
            raise ValueError('Non-finite run event value: ' + value)
        if not isinstance(json.loads(payload, parse_constant=invalid_constant), dict):
            raise ValueError('Run event payload must be an object')


def export_run_events(bundle_dir, output_path=None):
    """Read canonical records without rewriting either DB; preserve precision."""
    root = Path(bundle_dir).resolve()
    target = Path(output_path) if output_path is not None else root / 'run_events.jsonl'
    with sqlite3.connect((root / 'summary.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        if not has_event_streams(db):
            return None
        validate_event_streams(db)
        rows = db.execute('''SELECT e.run_uid, r.run_index, r.problem_seed, r.algorithm_seed,
            s.stream, s.schema_version, e.event_index, e.payload_json
            FROM run_event_records AS e
            JOIN run_event_streams AS s ON s.run_uid=e.run_uid AND s.stream=e.stream
            JOIN runs AS r ON r.run_uid=e.run_uid
            ORDER BY e.run_uid, s.stream, e.event_index''')
        temporary = target.with_name(target.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for uid, index, ps, als, stream, version, event_index, payload in rows:
                record = dict(run_uid=uid, run_index=index, problem_seed=ps,
                    algorithm_seed=als, schema=stream, schema_version=version,
                    event_index=event_index, data=json.loads(payload))
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(target)
    return target
