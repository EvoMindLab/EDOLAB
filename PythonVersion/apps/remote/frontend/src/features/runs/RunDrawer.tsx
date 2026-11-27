import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertOctagon, Check, ChevronRight, Circle, Clipboard, Cpu, FileArchive, GitBranch, HardDrive, PencilLine, RefreshCw, Square, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { createPolling } from "../../lib/polling";
import { api } from "../../api/client";
import { IconButton, Button, ConfirmDialog, Dialog, LoadingBlock, Meter, Notice, RunStatusPill, Tabs } from "../../components/ui";
import { formatBytes, formatDateTime, formatInteger, isTerminalRun, phaseLabel, shortSha } from "../../lib/format";
import type { ResourceSample, RunDetail, RunEvent, RunPhase, RunStage } from "../../types";
import { tr } from "../../lib/i18n";

const phases: RunPhase[] = ["queued", "preflight", "snapshot", "sync", "remote_setup", "execute", "retrieve", "verify", "cleanup", "complete"];
const syncStages = [
  { code: "sync_prepare_copy", label: tr("Preparing a separate copy") },
  { code: "sync_diff", label: tr("Computing code differences") },
  { code: "sync_upload", label: tr("Uploading code") },
  { code: "sync_verify", label: tr("Validating the execution copy") },
] as const;
const MAX_EVENT_MEMORY = 4_000;
const MAX_RENDERED_EVENTS = 1_500;
const MAX_SAMPLE_MEMORY = 2_000;

interface RunEvidence {
  detail: RunDetail;
  events: RunEvent[];
  samples: ResourceSample[];
}

type PhaseState = "pending" | "current" | "complete" | "failed";

function PhaseEntry({ label, state, detail, syncStep = false }: { label: string; state: PhaseState; detail?: string | null; syncStep?: boolean }) {
  const className = [
    syncStep ? "phase-list__sync-step" : "",
    state === "complete" ? "is-complete" : state === "current" ? "is-current" : state === "failed" ? "is-failed" : "",
  ].filter(Boolean).join(" ");
  return (
    <div className={className}>
      {state === "complete" ? <Check size={13} /> : state === "failed" ? <AlertOctagon size={13} /> : <Circle size={10} />}
      <span className="phase-list__copy"><span>{label}</span>{detail ? <small>{detail}</small> : null}</span>
      {state === "current" || state === "failed" ? <ChevronRight size={14} /> : null}
    </div>
  );
}

function syncStageOrdinal(stage: RunStage): number {
  const supplied = Number.isFinite(stage.ordinal) ? Math.trunc(stage.ordinal) : 0;
  if (supplied >= 1 && supplied <= syncStages.length) return supplied;
  const matched = syncStages.findIndex((item) => item.code === stage.code);
  return matched >= 0 ? matched + 1 : 1;
}

function syncModeLabel(mode: RunStage["mode"]): string | null {
  if (mode === "cache") return tr("Cache reused");
  if (mode === "delta") return tr("Incremental upload");
  if (mode === "full") return tr("Full upload");
  return null;
}

function syncProgressLabel(progress: RunStage["progress"]): string | null {
  if (!progress || !Number.isFinite(progress.current) || !Number.isFinite(progress.total)) return null;
  const current = Math.max(0, progress.current);
  const total = Math.max(0, progress.total);
  if (progress.unit === "bytes") return `${formatBytes(current)} / ${formatBytes(total)}`;
  return tr("{0} / {1} files", formatInteger(Math.round(current)), formatInteger(Math.round(total)));
}

function syncStepState(stage: RunStage, ordinal: number): PhaseState {
  const currentOrdinal = syncStageOrdinal(stage);
  if (ordinal < currentOrdinal) return "complete";
  if (ordinal > currentOrdinal) return "pending";
  if (stage.state === "completed") return "complete";
  if (stage.state === "failed") return "failed";
  return "current";
}

function syncStepDetail(stage: RunStage, ordinal: number): string | null {
  const parts: string[] = [];
  const currentOrdinal = syncStageOrdinal(stage);
  if (ordinal === 3) {
    const mode = syncModeLabel(stage.mode);
    if (mode) parts.push(tr("Mode: {0}", mode));
  }
  if (ordinal === currentOrdinal) {
    const count = Number.isFinite(stage.count) && stage.count > 0 ? Math.trunc(stage.count) : syncStages.length;
    parts.push(tr("Step {0}/{1}", currentOrdinal, count));
    const progress = syncProgressLabel(stage.progress);
    if (progress) parts.push(progress);
    if (stage.message.trim()) parts.push(stage.message.trim());
  }
  return parts.length ? parts.join(" · ") : null;
}

function mergeEvents(current: RunEvent[], incoming: RunEvent[]): RunEvent[] {
  const merged = new Map<string, RunEvent>();
  for (const event of [...current, ...incoming]) {
    merged.set(event.sequence == null ? "id:" + event.id : "sequence:" + event.sequence, event);
  }
  return [...merged.values()].sort((left, right) => {
    if (left.sequence != null && right.sequence != null) return left.sequence - right.sequence;
    return left.timestamp.localeCompare(right.timestamp);
  }).slice(-MAX_EVENT_MEMORY);
}

function mergeSamples(current: ResourceSample[], incoming: ResourceSample[]): ResourceSample[] {
  const merged = new Map<string, ResourceSample>();
  for (const sample of [...current, ...incoming]) {
    merged.set(sample.id == null ? "time:" + sample.timestamp : "id:" + sample.id, sample);
  }
  return [...merged.values()].sort((left, right) => {
    if (left.id != null && right.id != null) return left.id - right.id;
    return left.timestamp.localeCompare(right.timestamp);
  }).slice(-MAX_SAMPLE_MEMORY);
}

function useRunEvidence(runId: string | null, pollSeconds: number, refreshNonce: number) {
  const [data, setData] = useState<RunEvidence | null>(null);
  const [loading, setLoading] = useState(Boolean(runId));
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const refreshCurrent = useRef<(() => Promise<void>) | null>(null);
  const refresh = useCallback(() => refreshCurrent.current?.() ?? Promise.resolve(), []);

  useEffect(() => {
    setData(null);
    setError(null);
    setLoading(Boolean(runId));
    if (!runId) return;
    let active = true;
    let sequence = 0;
    let sampleId: number | undefined;
    let interval = pollSeconds * 1000;
    const polling = createPolling(async () => {
      setRefreshing(true);
      try {
        const [detail, events, samples] = await Promise.all([
          api.runs.get(runId),
          api.runs.events(runId, { afterSequence: sequence }),
          api.runs.samples(runId, { afterId: sampleId }),
        ]);
        if (!active) return;
        setData((previous) => ({ detail,
          events: mergeEvents(previous?.events ?? [], events),
          samples: mergeSamples(previous?.samples ?? [], samples),
        }));
        sequence = Math.max(sequence, ...events.map((event) => event.sequence ?? 0));
        sampleId = samples.reduce<number | undefined>((latest, sample) => sample.id == null ? latest : Math.max(latest ?? 0, sample.id), sampleId);
        // Terminal records can still be cleaned up or recovered externally.
        // Keep a slow refresh once evidence is drained and cleanup has settled.
        const settled = isTerminalRun(detail.status)
          && !["pending", "running"].includes(detail.remote_cleanup?.status ?? "skipped")
          && events.length < 1500 && samples.length < 1000;
        interval = Math.max(pollSeconds, settled ? 30 : 1) * 1000;
        setError(null);
      } catch (error) {
        if (active) setError(error instanceof Error ? error : new Error(tr("Could not read the run details")));
      } finally {
        if (active) { setLoading(false); setRefreshing(false); }
      }
    }, () => interval);
    refreshCurrent.current = polling.refresh;
    return () => { active = false; polling.dispose(); refreshCurrent.current = null; };
  }, [runId, pollSeconds]);
  useEffect(() => { void refresh(); }, [refresh, refreshNonce]);

  return { data, error, loading, refreshing, refresh };
}

export function RunDrawer({ runId, pollSeconds, refreshNonce, onClose, onChanged, onDeleted }: { runId: string | null; pollSeconds: number; refreshNonce: number; onClose: () => void; onChanged: () => void; onDeleted: () => void }) {
  const { data, error, loading, refreshing, refresh } = useRunEvidence(runId, pollSeconds, refreshNonce);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [confirmRecover, setConfirmRecover] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameName, setRenameName] = useState("");
  const [acting, setActing] = useState(false);
  const [eventFilter, setEventFilter] = useState<"all" | "stdout" | "stderr" | "system">("all");

  const filteredEvents = useMemo(() => data?.events.filter((event) => eventFilter === "all" || event.stream === eventFilter) ?? [], [data?.events, eventFilter]);
  const displayedEvents = useMemo(() => filteredEvents.slice(-MAX_RENDERED_EVENTS), [filteredEvents]);
  const currentPhaseIndex = data ? phases.indexOf(data.detail.phase) : 0;
  const syncStage = data?.detail.stage && (data.detail.stage.phase === "sync" || data.detail.stage.code.startsWith("sync_")) ? data.detail.stage : null;
  const syncStageFinished = Boolean(syncStage && syncStage.state === "completed" && syncStageOrdinal(syncStage) >= syncStages.length);
  const effectivePhaseIndex = syncStage && !syncStageFinished ? phases.indexOf("sync") : currentPhaseIndex;
  const syncStageHeading = syncStage && !syncStageFinished
    ? `${syncStages[syncStageOrdinal(syncStage) - 1]?.label ?? tr("Code sync")} · ${syncStageOrdinal(syncStage)}/${Number.isFinite(syncStage.count) && syncStage.count > 0 ? Math.trunc(syncStage.count) : syncStages.length}`
    : null;
  const runProgressMessage = data
    ? syncStage && !syncStageFinished
      ? tr("Experiment progress is counted separately; the code sync details are in the execution stages below.")
      : data.detail.current_message ?? phaseLabel[data.detail.phase]
    : "";
  const latestSample = data?.samples.at(-1);
  const terminal = Boolean(data && isTerminalRun(data.detail.status));
  const cleanupBlocksDelete = Boolean(
    data
    && terminal
    && data.detail.remote_workspace
    && data.detail.remote_cleanup
    && ["pending", "running", "failed"].includes(data.detail.remote_cleanup.status),
  );
  const deletable = terminal && !cleanupBlocksDelete;
  const recoverable = Boolean(terminal && data?.detail.recovery?.available);
  const recoveryMode = data?.detail.recovery?.mode;
  const recoveryLabel = recoveryMode === "cleanup"
    ? tr("Retry cleaning the execution folder")
    : recoveryMode === "verify"
      ? tr("Validate results again")
      : tr("Resume result collection");
  const recoveryTitle = recoveryMode === "cleanup"
    ? tr("Retry cleaning the execution folder now?")
    : recoveryMode === "verify"
      ? tr("Validate and finish this run again?")
      : tr("Resume this run?");
  const recoveryMessage = recoveryMode === "cleanup"
    ? tr("This only reconnects to the original machine and deletes the execution folder whose ownership is confirmed exactly by the marker of this record; it is offered only when the results have been collected and validated completely, or when the task can be shown never to have started. The experiment is not rerun, local results are not overwritten or deleted, and no new run record is created.")
    : recoveryMode === "verify"
      ? tr("Relay reuses the results already collected on this computer and repeats the integrity and source validation; the experiment is not rerun. The temporary folder on the executing machine is cleaned after the validation passes.")
      : data?.detail.recovery?.message ?? tr("Relay continues collecting results with the code, task, machine identity and execution folder frozen for this run; no new run record is created.");

  if (!runId) return null;

  const cancel = async () => {
    setActing(true);
    try {
      await api.runs.cancel(runId);
      toast.success(tr("Cancellation requested"), { description: tr("The controller terminates the execution processes and collects logs and results where it can; the audit events give the final state.") });
      setConfirmCancel(false);
      await refresh();
      onChanged();
    } catch (error) { toast.error(tr("Cancellation failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setActing(false); }
  };

  const remove = async () => {
    if (!data || !isTerminalRun(data.detail.status) || cleanupBlocksDelete) {
      toast.error(tr("This record cannot be deleted"), { description: cleanupBlocksDelete ? tr("The working folder on the executing machine has not been cleaned safely yet; Relay keeps the record so that it can retry.") : tr("Active runs cannot be deleted; cancel first and wait for a final state.") });
      return;
    }
    setActing(true);
    try {
      await api.runs.remove(runId);
      toast.success(tr("Run record and candidate source deleted"), { description: tr("Result copies that the user imported explicitly are not affected.") });
      setConfirmDelete(false);
      onDeleted();
    } catch (error) { toast.error(tr("Deletion failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setActing(false); }
  };

  const recover = async () => {
    if (!data?.detail.recovery?.available) return;
    setActing(true);
    try {
      await api.runs.recover(runId);
      toast.success(tr("Recovery started"), { description: recoveryMessage });
      setConfirmRecover(false);
      await refresh();
      onChanged();
    } catch (error) { toast.error(tr("Recovery failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setActing(false); }
  };

  const rename = async () => {
    const nextName = renameName.trim();
    if (!nextName) return;
    setActing(true);
    try {
      await api.runs.rename(runId, nextName);
      toast.success(tr("Run name updated"), { description: tr("Affects only the run record; the names of results already imported are not changed.") });
      setRenameOpen(false);
      await refresh();
      onChanged();
    } catch (error) { toast.error(tr("Rename failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setActing(false); }
  };

  return (
    <div className="drawer-layer" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <aside className="run-drawer" aria-label={tr("Run details")}>
        <header className="run-drawer__header">
          <div><h2>{data?.detail.task_label ?? data?.detail.task_path ?? runId}</h2><p>{runId}</p></div>
          <div className="run-drawer__header-actions">
            {data ? <IconButton aria-label={tr("Rename task")} title={tr("Rename task")} onClick={() => { setRenameName(data.detail.task_label ?? data.detail.task_path); setRenameOpen(true); }}><PencilLine size={16} /></IconButton> : null}
            <IconButton aria-label={tr("Close run details")} onClick={onClose}><X size={18} /></IconButton>
          </div>
        </header>

        <div className="run-drawer__body">
          {loading && !data ? <LoadingBlock label={tr("Reading the evidence of the run…")} /> : data ? (
            <>
              {error ? <Notice tone="danger"><strong>{tr("Live update failed")}</strong><p>{error.message}</p></Notice> : null}
              <section className="drawer-summary">
                <div className="drawer-summary__status"><RunStatusPill status={data.detail.status} /><span>{data.detail.server_name}</span><button onClick={() => void refresh()} aria-label={tr("Refresh details")}><RefreshCw size={14} className={refreshing ? "is-spinning" : ""} /></button></div>
                <span className="drawer-summary__progress-label">{tr("Experiment progress")}</span>
                <strong>{Math.round(data.detail.progress_percent)}%</strong>
                <p>{runProgressMessage}</p>
                <div className="progress-line progress-line--large"><span style={{ width: data.detail.progress_percent + "%" }} /></div>
                <small>{tr("{0} / {1} run units · sampled every {2} s", data.detail.completed_units, data.detail.total_units, data.detail.refresh_interval_seconds)}</small>
              </section>

              <section className="drawer-section">
                <header><h3>{tr("Execution stages")}</h3><span>{syncStageHeading ?? phaseLabel[data.detail.phase]}</span></header>
                <div className="phase-list">
                  {phases.map((phase, index) => {
                    if (phase === "sync" && syncStage) {
                      return syncStages.map((item, syncIndex) => (
                        <PhaseEntry
                          key={item.code}
                          label={item.label}
                          state={syncStepState(syncStage, syncIndex + 1)}
                          detail={syncStepDetail(syncStage, syncIndex + 1)}
                          syncStep
                        />
                      ));
                    }
                    const complete = data.detail.status === "succeeded" || index < effectivePhaseIndex;
                    const current = index === effectivePhaseIndex && data.detail.status !== "succeeded";
                    const state: PhaseState = complete ? "complete" : current && data.detail.status === "failed" ? "failed" : current ? "current" : "pending";
                    return <PhaseEntry key={phase} label={phaseLabel[phase]} state={state} />;
                  })}
                </div>
              </section>

              <section className="drawer-section">
                <header><h3>{tr("Live resources")}</h3><span>{latestSample ? formatDateTime(latestSample.timestamp) : tr("No samples yet")}</span></header>
                {latestSample ? <div className="resource-now"><Meter value={latestSample.cpu_percent} label="CPU" detail={latestSample.load_1m != null ? "load " + latestSample.load_1m.toFixed(2) : undefined} /><Meter value={latestSample.memory_percent} label={tr("Memory")} />{latestSample.gpus.map((gpu) => <Meter key={gpu.index} value={gpu.utilization_percent} label={"GPU " + gpu.index} detail={Math.round(gpu.memory_used_mb / 1024) + " / " + Math.round(gpu.memory_total_mb / 1024) + " GB"} />)}</div> : <p className="muted-copy">{tr("Resource samples are recorded at the refresh interval once the environment check passes and execution starts.")}</p>}
                {data.samples.length ? <div className="sample-chart" aria-label={tr("Recent CPU samples")}>{data.samples.slice(-28).map((sample) => <i key={sample.id ?? sample.timestamp} style={{ height: Math.max(5, sample.cpu_percent) + "%" }} title={formatDateTime(sample.timestamp) + " · CPU " + sample.cpu_percent + "%"} />)}</div> : null}
              </section>

              <section className="drawer-section">
                <header><h3>{tr("Reproducibility evidence")}</h3><span>{data.detail.git?.dirty ? tr("Includes uncommitted changes") : tr("Git working tree clean")}</span></header>
                <dl className="evidence-list">
                  <div><dt><GitBranch size={14} />Git commit</dt><dd>{shortSha(data.detail.git?.commit)} · {data.detail.git?.branch ?? "detached"}</dd></div>
                  <div><dt><Clipboard size={14} />{tr("Working tree")}</dt><dd>{data.detail.git?.dirty ? tr("dirty · changes included in the snapshot") : "clean"}</dd></div>
                  <div><dt><HardDrive size={14} />Snapshot SHA</dt><dd title={data.detail.git?.snapshot_sha256}>{shortSha(data.detail.git?.snapshot_sha256)}</dd></div>
                  <div><dt><FileArchive size={14} />Task SHA</dt><dd title={data.detail.git?.task_sha256}>{shortSha(data.detail.git?.task_sha256)}</dd></div>
                  <div><dt><Square size={14} />{tr("Execution PID")}</dt><dd>{data.detail.pid ?? tr("Not started")}</dd></div>
                  <div><dt><Cpu size={14} />{tr("Parallelism")}</dt><dd>{data.detail.max_parallel_runs} workers</dd></div>
                </dl>
                {data.detail.environment_report ? <div className="environment-checks">{data.detail.environment_report.checks.map((check) => <div key={check.name} className={"is-" + check.status}><span>{check.status === "passed" ? <Check size={13} /> : <AlertOctagon size={13} />}</span><strong>{check.name}</strong><p>{check.detail}</p></div>)}</div> : null}
              </section>

              <section className="drawer-section drawer-section--events">
                <header><h3>{tr("stdout / stderr events")}</h3><span>{tr("Showing the last {0} / {1} in the current window", displayedEvents.length, filteredEvents.length)}</span></header>
                <Tabs variant="segmented" ariaLabel={tr("Event stream filter")} value={eventFilter} onChange={setEventFilter} items={[
                  { value: "all", label: tr("All") },
                  { value: "stdout", label: "stdout" },
                  { value: "stderr", label: "stderr" },
                  { value: "system", label: "system" },
                ]} />
                <p className="event-window-note">{tr("The browser keeps only a limited live window; all events are still written to the local SQLite database.")}</p>
                <div className="event-log">{displayedEvents.map((event) => <article key={event.id} className={"event event--" + event.stream + " event--" + event.level}><time>{formatDateTime(event.timestamp)}</time><span>{event.stream}</span><p>{event.message}</p></article>)}</div>
              </section>

              <section className="drawer-section">
                <header><h3>{tr("Result artifacts")}</h3><span>{tr("{0} files", data.detail.artifacts?.length ?? 0)}</span></header>
                {data.detail.artifacts?.length ? <div className="artifact-list">{data.detail.artifacts.map((artifact) => <div key={artifact.path}><span><FileArchive size={15} /></span><div><strong>{artifact.path}</strong><small>{artifact.kind} · {formatBytes(artifact.size_bytes)}</small></div><i className={artifact.verified ? "is-ok" : ""}>{artifact.verified ? tr("Validated") : artifact.transferred ? tr("Collected") : tr("To be collected")}</i></div>)}</div> : <p className="muted-copy">{tr("The complete results are collected item by item; if any step fails, its error context is kept.")}</p>}
                <dl className="result-facts">
                  <div><dt>{tr("Local results")}</dt><dd>{data.detail.local_result_path ?? tr("Not created yet")}</dd></div>
                  <div><dt>{tr("Exit code")}</dt><dd>{data.detail.exit_code ?? tr("Running")}</dd></div>
                  <div><dt>{tr("Error code")}</dt><dd>{data.detail.error_code ?? "—"}</dd></div>
                  <div><dt>{tr("Cleanup on the executing machine")}</dt><dd>{data.detail.remote_workspace ? data.detail.remote_cleanup?.status ?? "pending" : tr("No cleanup needed")}</dd></div>
                </dl>
                {data.detail.error_message ? <Notice tone="danger"><strong>{data.detail.error_code ?? tr("Run failed")}</strong><p>{data.detail.error_message}</p></Notice> : null}
                {data.detail.remote_workspace && data.detail.remote_cleanup?.error ? <Notice tone="warning"><strong>{tr("Execution folder cleanup not finished")}</strong><p>{data.detail.remote_cleanup.error}</p></Notice> : null}
                {recoverable ? <Notice tone="warning"><strong>{recoveryMode === "cleanup" ? tr("The execution folder of this record can be cleaned up again safely") : recoveryMode === "verify" ? tr("Results collected; they can be validated again") : tr("This run can be resumed safely")}</strong><p>{recoveryMessage}</p></Notice> : null}
                {data.detail.result_candidate_ready || data.detail.result_imported ? <div className="run-result-library-action"><div><strong>{data.detail.result_imported ? tr("This candidate already has a separate imported copy") : tr("The results are in the candidate list")}</strong><p>{data.detail.result_imported ? tr("The imported copy is managed separately by the result analysis module; deleting the run does not delete it.") : tr("The run module provides only the execution evidence and the candidate source. Import explicitly in “Analysis → Candidates”; until then the results cannot be previewed, plotted, exported or used in SA.")}</p></div></div> : null}
              </section>
            </>
          ) : <Notice tone="danger"><strong>{tr("Could not read the run details")}</strong><p>{error?.message ?? tr("Check that the local Relay service is still running.")}</p></Notice>}
        </div>

        {data ? <footer className="run-drawer__footer">
          {!isTerminalRun(data.detail.status) ? <Button tone="danger" onClick={() => setConfirmCancel(true)} icon={<Square size={14} />}>{tr("Cancel run")}</Button> : recoverable ? <Button tone="primary" onClick={() => setConfirmRecover(true)} icon={<RefreshCw size={14} />}>{recoveryLabel}</Button> : <span />}
          <div className="delete-run-action">
            {!terminal ? <small>{tr("Active runs cannot be deleted; cancel first and wait until they end.")}</small> : cleanupBlocksDelete ? <small>{recoverable ? tr("The data on the executing machine has not been wrapped up safely; use the recovery action first.") : tr("The execution folder cleanup is not finished; the record is kept so that it can be retried safely.")}</small> : null}
            <Button tone="quiet" disabled={!deletable} title={!terminal ? tr("Active runs cannot be deleted") : cleanupBlocksDelete ? tr("Waiting for the execution folder cleanup") : tr("Delete the run evidence and the candidate source")} onClick={() => { if (deletable) setConfirmDelete(true); }} icon={<Trash2 size={15} />}>{tr("Delete run record")}</Button>
          </div>
        </footer> : null}
      </aside>

      <ConfirmDialog open={confirmCancel} title={tr("Cancel this run?")} message={tr("The controller sends a termination signal to the separate process group and collects logs, state and partial or complete results where it can. What can be collected depends on the stage at which the run is cancelled; the exit code and the actual handling are written to the audit record.")} confirmLabel={tr("Confirm cancellation")} destructive busy={acting} onClose={() => setConfirmCancel(false)} onConfirm={() => void cancel()} />
      <ConfirmDialog open={confirmRecover && recoverable} title={recoveryTitle} message={recoveryMessage} confirmLabel={recoveryLabel} busy={acting} onClose={() => setConfirmRecover(false)} onConfirm={() => void recover()} />
      <ConfirmDialog open={confirmDelete && deletable} title={tr("Delete the run record and candidate source?")} message={tr("This deletes the evidence of the finished run and its candidate source. Result copies that the user imported explicitly are not deleted, and neither is the SA history. This cannot be undone.")} confirmLabel={tr("Delete run record permanently")} destructive busy={acting} onClose={() => setConfirmDelete(false)} onConfirm={() => void remove()} />
      <Dialog open={renameOpen} title={tr("Rename run record")} description={tr("Changes only the local display name of the run evidence; the frozen JSONL, code snapshot, result files and source hashes are unchanged, and imported results are not renamed.")} onClose={() => setRenameOpen(false)} footer={<><Button onClick={() => setRenameOpen(false)} disabled={acting}>{tr("Cancel")}</Button><Button tone="primary" onClick={() => void rename()} disabled={acting || !renameName.trim()}>{acting ? tr("Saving…") : tr("Save name")}</Button></>}>
        <label className="field"><span>{tr("Display name of the run")}</span><input autoFocus maxLength={200} value={renameName} onChange={(event) => setRenameName(event.target.value)} /><small>{tr("At most 200 characters; imported results and SA use their own frozen source names.")}</small></label>
      </Dialog>
    </div>
  );
}
