# EDOLAB Relay architecture

EDOLAB Relay is the local control plane for reproducible EDOLAB JSONL
experiments on this machine or SSH-accessible servers. Benchmark/NBN views and
result plots share the Web application, but run synchronously in the local
Relay process as transient interactive operations rather than experiments.
The retired Web workbench has been removed without changing Relay's experiment
records or recovery protocols.

## Product boundary

Relay owns:

- SSH server profiles and host-key trust;
- environment and admission checks for JSONL experiments;
- immutable source snapshots from the current working tree for JSONL experiments;
- durable JSONL experiment queues, lifecycle events, and resource samples;
- remote process launch, monitoring, cancellation, and restart recovery;
- verified experiment-artifact transfer and exact remote cleanup;
- read-only discovery of completed, verified batch outputs as result candidates;
- explicit import of a candidate or external standard bundle into an independent
  managed result library;
- imported-result previews, export, plotting, SA candidate selection, and frozen
  analysis history;
- bounded local interactive workspaces for Benchmark/NBN and result plots;
- a local Web UI, a Python launcher, and an optional Docker launcher.

Relay does not own algorithm lifecycle, FE accounting, metrics, or result
serialization.  Remote workers call the existing `ExperimentRunner` and write
the existing `summary.sqlite` plus `trend.sqlite` bundle.

Only JSONL algorithm/problem experiments choose a local or SSH execution target
and create source snapshots, admission checks, queue entries, database run
records, or recoverable workers. Benchmark/NBN and result plotting call existing
functions directly in the Relay process and return in the initiating HTTP
request. They do not select a target, initiate SSH, or create a durable run.

Run completion does not grant analysis access. A completed, checksum-verified
batch run is only a result candidate. The user must explicitly import that
candidate, or import an external standard bundle, before its runsets can be
previewed, exported, plotted, or added to SA. The managed imported copy has an
independent lifecycle from the source run record.

## Deployment shape

```text
browser
   |
   v
FastAPI control plane (localhost only)
   |-- SQLite: servers, experiment/legacy runs, events, samples, artifacts, settings
   |-- JSONL experiment artifacts: exact source/task/provenance/log/result files
   |-- JSONL run scheduler: at most one active snapshot per server
   |-- resource scheduler: bounded background SSH sampling -> SQLite history
   |-- interactive Benchmark/NBN -> local function -> HTTP result
   |-- interactive plots -> verified local bundle -> local function -> HTTP result
   |      `-- transient files: analyses/interactive/{benchmarks,plots}
   `-- system OpenSSH client (the private key never leaves this machine)
              |
              v
       remote JSONL-experiment workspace
       |-- immutable verified source
       |-- detached process group and logs
       `-- result bundle (deleted only after verified local receipt)
```

The Python and Docker launchers use the same schema and artifact layout.  A
controller lock prevents them from writing the SQLite workspace concurrently.

## Durable state

The default data root is `results/remote_control/`:

```text
remote_control/
  control.sqlite
  control.lock
  known_hosts
  ssh_mux/
  analyses/
    results/<result-id>/
    interactive/
      benchmarks/<uuid>/
      plots/<uuid>/
  working/<run-id>/
  runs/<run-id>/
  trash/
```

SQLite uses WAL mode, foreign keys, a busy timeout, and short transactions.
Run events and run-bound resource samples are append-only. General monitoring
samples are append-only until the configured retention job prunes them. Key
material is never stored: server rows contain only the local identity-file path.

The result-analysis data flow is deliberately import-driven:

```text
completed verified batch run -> candidate (run-owned, no analysis access)
candidate or external bundle -> explicit import -> analyses/results/<result-id>/
imported result -> runset selection -> current SA candidates -> frozen SA history
```

`result_imports` records the independent managed copies. Current `sa_imports`
must reference those imported results; they are not synthesized from ordinary
completed runs. Deleting a run removes its candidate source but leaves an
already imported copy intact. Deleting an imported result removes its current
SA candidates, while completed analyses retain their frozen source snapshots.

`analyses/interactive/benchmarks/` and `analyses/interactive/plots/` are
transient caches, not a job ledger or experiment archive. Each request that
needs files receives a UUID directory. Each category independently retains at
most the newest 128 managed directories; creating a new directory prunes the
oldest excess directories. In-memory surface and NBN models create no artifact
directory. Interactive operations do not add rows to the run, event, artifact,
or resource-sample tables.

General server-resource samples are retained for 72 hours by default and are
pruned independently from run-bound resource evidence. Both retention and the
default three-second sampling interval are settings. Disabling a server only
removes it from new-run scheduling; a trusted server continues to be sampled.
Sampling is paused until a successful full probe has pinned the SSH host-key
fingerprint. Samples also carry a hash of the SSH target identity, so editing a
server profile cannot make load history from its previous target appear as the
current machine's history.

Each general sample also carries an internal digest of the connection target
(host, port, account, key path, remote root, and pinned fingerprint). If a
server record is repointed, old samples remain durable but are excluded from
the new target's latest/history views. An in-flight attempt rechecks that
identity before persisting, so metrics from two machines cannot be mixed under
one visible server history.

Deleting a terminal run first moves its artifact directory to local trash,
then deletes the database row in a transaction.  A failed database deletion
restores the directory; trash left by a crash is safe to reap later.

## JSONL experiment run state machine

```text
queued -> checking -> snapshotting -> syncing -> preparing -> running
       -> transferring -> verifying -> cleaning -> completed

canceling -> canceled
any phase -> failed
controller restart -> interrupted -> resume/reconcile or failed
terminal transfer/launch failure -> explicit resume -> running/transferring
terminal cleanup failure -> automatic or explicit cleanup retry
```

`completed` means all requested result files are present locally, their
manifest hashes match, and both SQLite databases pass `PRAGMA quick_check`.
Normally the exact remote directory is removed in the same pipeline.  If that
last deletion fails, the scientifically successful run remains `completed` with
an explicit cleanup-pending state and is retried without risking the local
result.

This state machine does not apply to Benchmark/NBN or result plotting. Those
operations either return their result in the current request or return an HTTP
error; they have no queued, running, recovery, or terminal database state.

## Reproducible source synchronization for JSONL experiments

Git commit identifiers are descriptive, not authoritative: uncommitted changes
must also run.  Relay builds a snapshot from tracked files and non-ignored
untracked files in the runtime surface (`core`, `instances`, `utility`, the
relevant `apps` and scripts, `cli.py`, and `pyproject.toml`).  Generated output,
tests, test-only `*_reference.mat` fixtures, caches, and
repository metadata are excluded. Runtime model checkpoints and their config
files remain included.

The snapshot records each relative path, byte size, executable mode, and
SHA-256.  Its canonical manifest SHA-256 is the execution identity.  Relay also
records the branch, commit, dirty state, and diff digest, and retains the exact
source archive with the local run.

Run input may come from a repository `tasks/*.jsonl` file or directly pasted
JSONL. Inline input is parsed from its exact UTF-8 bytes and is never written to
the repository `tasks/` directory. Relay freezes the original content, its
SHA-256, and the parsed task payload before queueing, so inspection, launch, and
recovery all refer to the same immutable input.

Before launch Relay compares the new manifest with a verified remote cache and
records added, content-changed, metadata-only, and deleted files. Only added or
content-changed bytes enter `source-delta.tar.gz`; deletions, mode changes, and
Git-provenance-only changes require no source payload. An exact snapshot cache
hit uploads no source bytes.

The v2 remote cache stores sealed, read-only snapshot trees, while every run
materializes its own writable `runs/<run-id>/source/` directory with independent
file inodes. Tree cloning prefers filesystem reflink/COW and safely falls back
to a normal copy. Changed payload members are checked against their manifest
digests, cache trees are guarded by an exact file-set and stat seal, and cache
publication is atomic. Normal cache hits and small deltas therefore do not hash,
extract, or rebuild a complete source archive. Legacy v1 archive caches are
fully verified and promoted once on first reuse. A run never executes from a
mutable shared checkout. The durable UI stages separate remote-copy
preparation, manifest comparison, source upload/cache reuse, and final
materialization/verification.

The interactive Benchmark/NBN and result-plot routes do not build, consult, or
transfer a source snapshot. They execute the already-running Relay installation
and therefore are intentionally outside experiment provenance.

The tree stat seal detects accidental mutation and binds validation to the
file descriptors used for cloning, but it is intentionally not an
authentication boundary against code running as the same Unix account. Relay
therefore treats uploaded scientific code as trusted and keeps its cache under
the Relay account's private root. Adversarial workloads must use a separate OS
account or host rather than sharing this cache.

## Environment and admission checks for JSONL experiments

A connection probe distinguishes installed commands from usable facilities.
For example, an installed `nvidia-smi` whose driver query fails means GPU is
unavailable.  Checks include:

- SSH public-key authentication and a private `known_hosts` file. The first
  environment check uses OpenSSH `accept-new` TOFU, displays the full resulting
  SHA-256 fingerprint for independent verification, and every later host-key
  change is rejected;
- remote-root ownership marker, writability, free bytes, and free inodes;
- OS, architecture, logical/physical CPUs, load, memory, and disk;
- GPU name, utilization, memory, driver, and actual usability;
- Conda availability (including common non-interactive `/opt` installations),
  the `EDOLAB` environment, Python version, and registry. The successful probe
  records the resolved absolute Conda executable used by that run;
- rsync/tar/git and Docker availability versus permission to use them.

Default admission is conservative: one active Relay run per server, one
parallel EDOLAB run inside a task, sufficient disk, and a compatible `EDOLAB`
environment. Server load is a visible non-blocking warning rather than an
admission failure: transient contention should inform scheduling without making
the machine unusable. Explicit overrides for the remaining overrideable limits
are recorded as audit events. Relay never invokes `sudo`.

The server inventory renders every probe condition independently instead of
collapsing unrelated failures into one environment result. A bounded background
scheduler runs the lightweight CPU, load, memory, and GPU command independently
of browser visibility, stores both successes and failed attempts in SQLite, and
backs off after transport failures. Dashboard and server-page polling perform
read-only local database queries; they never initiate SSH. An explicit
cross-site-protected POST remains available only for a deliberate “sample now”
operation. The lightweight sampler never replaces the durable full probe.

Interactive Benchmark/NBN and result plotting do not choose a compute target,
so they do not invoke these probes or admission rules. Their request schemas and
workspace limits still validate input size and shape to keep the local Relay
process bounded.

The one-active-snapshot rule is essential because the mandatory
`pip install -e .` changes the shared Conda environment's editable source path.
Different snapshots may run concurrently only after a future isolated
environment or container executor is selected.

## JSONL remote execution and recovery

The remote command reuses the exact Conda executable frozen by the fresh
execution probe, activates `EDOLAB`, performs the repository's required editable
install and registry check, then launches `apps.remote.worker` in a detached
process group. It never performs a second, potentially divergent Conda search
during launch or recovery. The worker calls `ExperimentRunner` directly so the
output directory and maximum parallelism are exact, and emits structured JSON
after every complete run.

The local controller polls a small remote state file plus stdout/stderr offsets
at the run's configured progress interval. The independent resource scheduler
uses its own configurable sampling interval and persists utilization history in
SQLite. A dropped browser or local controller does not define job lifetime. On
restart, persisted remote PID, directory, offsets, and state are reconciled and
monitoring resumes.

Transport loss is retried without terminalizing a detached job.  Deterministic
checksum, remote-protocol, and local-filesystem failures stop instead of looping
forever, but they do not strand the remote result: the run exposes an explicit
recovery action that reuses its immutable snapshot, frozen task, frozen server
identity, PID/launch intent, logs, and marker-owned directory.  A pre-launch
workspace is eligible for automatic cleanup only when durable state proves that
no launch intent was ever persisted.  Once launch may have begun, Relay first
recovers the idempotent process identity and never guesses that deletion is safe.

Current core APIs expose progress after each complete run, not within a run.
Relay therefore shows a heartbeat and resource usage during a long individual
run; FE-level progress requires a later stable observer in `core`.

Cancellation targets the dedicated remote process group.  It first sends
`TERM`, waits for a grace period, then may send `KILL`.  The exact signal and
exit code are recorded.  Partial results and logs are collected when possible.

## JSONL experiment artifact transfer and cleanup

After the worker closes its databases, the remote collector runs SQLite quick
checks and writes a SHA-256 manifest for results, logs, status, environment, and
provenance.  Transfer follows this order:

1. download into a local partial directory;
2. reject unsafe archive paths;
3. verify the complete manifest and SQLite bundle;
4. commit per-file artifact rows and durable transfer-verification state;
5. repeat the complete local checksum/bundle pass immediately before deletion;
6. request deletion of the exact marker-owned remote run directory and verify it;
7. atomically promote the local working directory to its final run location.

If any download or verification step fails, remote artifacts are retained and
the UI offers recovery of the same run.  Cleanup never accepts `/`, `$HOME`, an
unresolved variable, a glob, or an unmarked path as a target.

## Historical Benchmark and plot runs

Relay releases before the local-interactive boundary could create durable
`benchmark` and `plot` run rows. Existing rows and their verified artifacts
remain readable, downloadable, and deletable through compatibility readers, so
an upgrade does not destroy history. New Web/API requests use only the direct
local render routes and cannot create either legacy run kind. Compatibility
readers do not turn transient files under `analyses/interactive/` into run
records and do not enqueue new work.

## Authorized test-server finding (2026-08-10)

Connection details are intentionally redacted from publishable source. The
authorized server successfully exercised SSH-key authentication, TOFU pinning,
resource probing, and failure-path reporting. Remote load was excessive and was
reported as a non-blocking scheduling warning. The server was not admitted for a
scientific run because no usable NVIDIA GPU was available, configured storage
was unsuitable, and Python 3.11 with Conda `EDOLAB` was absent.

This is the intended safety result. Relay must expose the failed conditions and
refuse execution instead of silently falling back to system Python or an
unapproved directory. A compliant environment and writable dedicated storage
must be provisioned before that target can run experiments.
