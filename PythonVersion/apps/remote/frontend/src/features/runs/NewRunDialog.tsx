import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { AlertTriangle, Calculator, CheckCircle2, FileJson2, FolderOpen, Laptop, Server, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import type { CreateRunInput, RemoteServer, RemoteSettings, TaskFile, TaskInspection } from "../../types";
import { formatInteger } from "../../lib/format";
import { api } from "../../api/client";
import { Button, Dialog, Notice, Tabs } from "../../components/ui";
import { tr } from "../../lib/i18n";

type TaskSourceMode = "file" | "jsonl";
const MAX_INLINE_TASK_BYTES = 2 * 1024 * 1024;

interface JsonlSyntaxCheck {
  objectCount: number;
  error: string | null;
}

const checkJsonlSyntax = (source: string): JsonlSyntaxCheck => {
  if (!source.trim()) return { objectCount: 0, error: tr("Paste at least one JSONL task line.") };
  let objectCount = 0;
  for (const [index, rawLine] of source.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch (error) {
      const detail = error instanceof Error ? error.message : tr("Invalid JSON");
      return { objectCount, error: tr("Line {0} is not valid JSON: {1}", index + 1, detail) };
    }
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return { objectCount, error: tr("Line {0} must be a JSON object.", index + 1) };
    }
    objectCount += 1;
  }
  if (!objectCount) return { objectCount: 0, error: tr("The JSONL holds no JSON object that can be parsed; blank lines and # comments are ignored.") };
  return { objectCount, error: null };
};

export function NewRunDialog({ open, initialTaskPath, tasks, servers, settings, onClose, onCreated }: { open: boolean; initialTaskPath?: string | null; tasks: TaskFile[]; servers: RemoteServer[]; settings: RemoteSettings; onClose: () => void; onCreated: (runId: string) => void }) {
  const selectableServers = useMemo(() => servers.filter((server) => server.enabled), [servers]);
  const [taskSourceMode, setTaskSourceMode] = useState<TaskSourceMode>("file");
  const [form, setForm] = useState<CreateRunInput>({
    display_name: "",
    task_path: "",
    task_jsonl: "",
    server_id: "",
    max_parallel_runs: settings.default_max_parallel_runs,
    refresh_interval_seconds: settings.default_poll_interval_seconds,
    admission_override: false,
  });
  const [inspection, setInspection] = useState<TaskInspection | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [creating, setCreating] = useState(false);
  const wasOpen = useRef(false);

  useEffect(() => {
    if (!open) {
      wasOpen.current = false;
      return;
    }
    if (wasOpen.current) return;
    wasOpen.current = true;
    setTaskSourceMode(tasks.length ? "file" : "jsonl");
    setForm({
      display_name: "",
      task_path: tasks.some((task) => task.path === initialTaskPath) ? initialTaskPath ?? "" : tasks[0]?.path ?? "",
      task_jsonl: "",
      server_id: selectableServers.find((server) => server.status === "online")?.id ?? selectableServers[0]?.id ?? "",
      max_parallel_runs: settings.default_max_parallel_runs,
      refresh_interval_seconds: settings.default_poll_interval_seconds,
      admission_override: false,
    });
    setInspection(null);
    setInspectError(null);
  }, [initialTaskPath, open, selectableServers, settings.default_max_parallel_runs, settings.default_poll_interval_seconds, tasks]);

  const jsonlSyntax = useMemo(() => checkJsonlSyntax(form.task_jsonl ?? ""), [form.task_jsonl]);
  const jsonlByteCount = useMemo(() => new TextEncoder().encode(form.task_jsonl ?? "").byteLength, [form.task_jsonl]);
  const jsonlSizeError = jsonlByteCount > MAX_INLINE_TASK_BYTES
    ? tr("The JSONL is {0} bytes in UTF-8, more than the limit of {1} bytes.", formatInteger(jsonlByteCount), formatInteger(MAX_INLINE_TASK_BYTES))
    : null;
  const requestInput = useMemo<CreateRunInput>(() => ({
    ...(form.display_name?.trim() ? { display_name: form.display_name.trim() } : {}),
    server_id: form.server_id,
    max_parallel_runs: form.max_parallel_runs,
    refresh_interval_seconds: form.refresh_interval_seconds,
    admission_override: form.admission_override,
    ...(taskSourceMode === "file"
      ? { task_path: form.task_path }
      : { task_jsonl: form.task_jsonl }),
  }), [form.admission_override, form.display_name, form.max_parallel_runs, form.refresh_interval_seconds, form.server_id, form.task_jsonl, form.task_path, taskSourceMode]);
  const inspectionInput = useMemo<CreateRunInput>(() => ({
    ...requestInput,
    admission_override: false,
  }), [form.max_parallel_runs, form.refresh_interval_seconds, form.server_id, form.task_jsonl, form.task_path, taskSourceMode]);

  useEffect(() => {
    const sourceReady = taskSourceMode === "file"
      ? Boolean(form.task_path)
      : Boolean(form.task_jsonl?.trim()) && !jsonlSyntax.error && !jsonlSizeError;
    if (!open || !sourceReady || !form.server_id || form.max_parallel_runs < 1) {
      setInspection(null);
      setInspectError(null);
      setInspecting(false);
      return;
    }
    setInspection(null);
    setInspectError(null);
    let active = true;
    const timer = window.setTimeout(async () => {
      setInspecting(true);
      setInspectError(null);
      try {
        const result = await api.tasks.inspect(inspectionInput);
        if (active) setInspection(result);
      } catch (error) {
        if (active) { setInspection(null); setInspectError(error instanceof Error ? error.message : tr("Task parsing failed")); }
      } finally { if (active) setInspecting(false); }
    }, 320);
    return () => { active = false; window.clearTimeout(timer); };
  }, [form.max_parallel_runs, form.server_id, form.task_jsonl, form.task_path, inspectionInput, jsonlSizeError, jsonlSyntax.error, open, taskSourceMode]);

  const selectedServer = selectableServers.find((server) => server.id === form.server_id);
  const hardBlockedReasons = inspection?.hard_blocked_reasons ?? [];
  const hasStructuredAdmission = inspection?.hard_blocked_reasons !== undefined || inspection?.overrideable_reasons !== undefined;
  const overrideableReasons = inspection?.overrideable_reasons ?? (hasStructuredAdmission ? [] : inspection?.blocked_reasons ?? []);
  const overrideAllowed = Boolean(inspection?.override_allowed && overrideableReasons.length && !hardBlockedReasons.length);
  const submissionBlocked = Boolean(
    inspection && (
      hardBlockedReasons.length
      || (!inspection.admissible && !(overrideAllowed && form.admission_override))
    ),
  );
  const localSourceError = taskSourceMode === "jsonl" ? (jsonlSizeError ?? jsonlSyntax.error) : null;

  const switchTaskSource = (mode: TaskSourceMode) => {
    setTaskSourceMode(mode);
    setForm((current) => ({ ...current, admission_override: false }));
    setInspection(null);
    setInspectError(null);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!inspection || submissionBlocked || localSourceError) return;
    setCreating(true);
    try {
      const run = await api.runs.create(requestInput);
      onCreated(run.id);
    } catch (error) {
      toast.error(tr("Run creation failed"), { description: error instanceof Error ? error.message : tr("Unknown error") });
    } finally { setCreating(false); }
  };

  return (
    <Dialog
      open={open}
      wide
      title={tr("New run")}
      description={tr("Choose this computer or an SSH server as the machine, then create a traceable run from a task file or pasted JSONL. The actual budget and the environment are checked before submission.")}
      onClose={onClose}
      footer={<><Button onClick={onClose} disabled={creating}>{tr("Cancel")}</Button><Button tone="primary" type="submit" form="new-run-form" disabled={creating || inspecting || !inspection || Boolean(localSourceError) || submissionBlocked}>{creating ? tr("Creating…") : tr("Confirm and queue")}</Button></>}
    >
      <form id="new-run-form" className="new-run-layout" onSubmit={(event) => void submit(event)}>
        <div className="new-run-fields">
          <label className="field field--wide"><span>{tr("Display name of the task (optional)")}</span><input maxLength={200} value={form.display_name ?? ""} placeholder={tr("For example: AMPPSO_GI paper main experiment · 2026-08")} onChange={(event) => setForm((current) => ({ ...current, display_name: event.target.value }))} /><small>{tr("The name used in the local run records and SA sources; it may differ from the JSONL file name and can be changed later.")}</small></label>

          <Tabs
            variant="segmented"
            className="field--wide ui-tabs--stretch"
            ariaLabel={tr("Task input")}
            value={taskSourceMode}
            onChange={switchTaskSource}
            items={[
              { value: "file", label: tr("Task file of the repository"), icon: <FolderOpen size={14} />, disabled: !tasks.length },
              { value: "jsonl", label: tr("Paste JSONL"), icon: <FileJson2 size={14} /> },
            ]}
          />

          {taskSourceMode === "file" ? (
            <label className="field field--wide"><span>{tr("Task file")}</span><select required value={form.task_path ?? ""} onChange={(event) => { setInspection(null); setInspectError(null); setForm((current) => ({ ...current, task_path: event.target.value, admission_override: false })); }}>{tasks.map((task) => <option key={task.path} value={task.path}>{task.path}</option>)}</select><small>{tr("Only tasks/*.jsonl of the repository are listed; the server reads the actual file again.")}</small></label>
          ) : (
            <label className="field field--wide jsonl-input"><span>{tr("Custom JSONL")}</span><textarea required spellCheck={false} value={form.task_jsonl ?? ""} aria-invalid={Boolean(localSourceError)} placeholder={'{"max_parallel_runs": 2, "default_problem": "GMPB"}\n{"label": "smoke", "algorithm": "ACFPSO", "runs": 1}'} onChange={(event) => { setInspection(null); setInspectError(null); setForm((current) => ({ ...current, task_jsonl: event.target.value, admission_override: false })); }} /><small className={localSourceError ? "field-hint field-hint--danger" : "field-hint field-hint--success"}>{localSourceError ? tr("{0} Currently {1} / {2} bytes.", localSourceError, formatInteger(jsonlByteCount), formatInteger(MAX_INLINE_TASK_BYTES)) : tr("Local syntax check passed: {0} JSON objects; {1} / {2} UTF-8 bytes. The server still checks the algorithms, problems, parameters and FE budget.", jsonlSyntax.objectCount, formatInteger(jsonlByteCount), formatInteger(MAX_INLINE_TASK_BYTES))}</small></label>
          )}

          <label className="field field--wide"><span>{tr("Machines")}</span><select required value={form.server_id} onChange={(event) => { setInspection(null); setInspectError(null); setForm((current) => ({ ...current, server_id: event.target.value, admission_override: false })); }}>{selectableServers.map((server) => <option key={server.id} value={server.id}>{server.target_kind === "local" ? tr("This computer") : "SSH"} · {server.name} · {server.status === "online" && server.fingerprint ? tr("Checked") : tr("Needs check")}</option>)}</select><small>{selectedServer ? (selectedServer.target_kind === "local" ? tr("Starts a separate background process from a frozen copy of the code, without SSH.") : selectedServer.username + "@" + selectedServer.host + ":" + selectedServer.port) : tr("No machine can be scheduled; enable this computer or add an SSH server first.")}</small></label>
          {selectedServer ? <div className="selected-server-trust field--wide"><strong>{selectedServer.target_kind === "local" ? <><Laptop size={14} />{" "}{tr("Local identity")}</> : tr("SSH fingerprint")}</strong><code>{selectedServer.fingerprint ?? tr("Not pinned yet")}</code><small>{selectedServer.target_kind === "local" ? tr("Runs on this computer also freeze the code and run in a separate folder; after a web restart, control is recovered by process identity.") : selectedServer.fingerprint ? tr("The first check pins the key (TOFU); make sure the full fingerprint matches a trusted source.") : tr("The environment check must succeed on the server page first; servers that were not probed cannot run.")}</small></div> : null}
          <label className="field"><span>{tr("Parallel runs")}</span><input type="number" min={1} max={256} value={form.max_parallel_runs} onChange={(event) => { setInspection(null); setInspectError(null); setForm((current) => ({ ...current, max_parallel_runs: Number(event.target.value), admission_override: false })); }} /><small>{tr("Number of run-level workers, not threads inside libraries.")}</small></label>
          <label className="field"><span>{tr("Run refresh interval (s)")}</span><input type="number" min={1} max={300} value={form.refresh_interval_seconds} onChange={(event) => setForm((current) => ({ ...current, refresh_interval_seconds: Number(event.target.value) }))} /><small>{tr("Controls only the progress and event polling of this run; resource sampling of the machines is set on the settings page.")}</small></label>
          <label className={"check-field field--wide" + (!overrideAllowed ? " check-field--disabled" : "")}><input type="checkbox" checked={form.admission_override && overrideAllowed} disabled={!overrideAllowed} onChange={(event) => setForm((current) => ({ ...current, admission_override: event.target.checked }))} /><span><strong>{tr("Admission override · resource limits that can be overridden only")}</strong><small>{overrideAllowed ? tr("Overrides are written to the audit; hard conditions such as machine identity, execution channel, folders, Conda and GPU can never be bypassed.") : tr("Nothing can be overridden in this check; hard environment errors must be fixed before submission.")}</small></span></label>
        </div>

        <aside className="inspection-card" aria-live="polite">
          <header><span><Calculator size={17} /></span><div><h3>{tr("Task budget check")}</h3><p>{inspecting ? tr("Being parsed by the server…") : inspection ? (taskSourceMode === "jsonl" ? tr("Computed from the pasted content") : tr("Computed from the current file")) : tr("Waiting for a valid task input")}</p></div></header>
          {localSourceError ? <Notice tone="danger"><strong>{tr("Local JSONL check")}</strong><p>{localSourceError}</p></Notice> : null}
          {inspectError ? <Notice tone="danger"><strong>{tr("Server task check")}</strong><p>{inspectError}</p></Notice> : null}
          {inspection ? (
            <>
              <div className="inspection-metrics">
                <span><small>{tr("Total runs")}</small><strong>{formatInteger(inspection.total_runs)}</strong></span>
                <span><small>{tr("Total FE")}</small><strong>{formatInteger(inspection.total_evaluations)}</strong></span>
                <span><small>{tr("Workers used")}</small><strong>{inspection.estimated_worker_slots}</strong></span>
                <span><small>{tr("Test problem")}</small><strong>{inspection.total_tasks}</strong></span>
              </div>
              <div className="inspection-facts">
                <span><Server size={14} />{inspection.problems.join(tr(", ")) || tr("Unrecognized problem")}</span>
                <span><ShieldCheck size={14} />{inspection.algorithms.length ? inspection.algorithms.map((name, index) => <span key={name}>{index ? tr(", ") : ""}{name}{inspection.temporary_algorithms?.[name] ? <em className="algorithm-temporary" title={inspection.temporary_algorithms[name]}>{tr("Temporary")}</em> : null}</span>) : tr("Unrecognized algorithm")}</span>
              </div>
              {inspection.entries.length ? <div className="inspection-entries">{inspection.entries.slice(0, 6).map((entry, index) => <div key={entry.label + "-" + index}><strong>{entry.label}</strong><span>{entry.run_count} runs × {formatInteger(entry.fe_per_run)} FE</span></div>)}</div> : null}
              {inspection.entries.length > 6 ? <small className="inspection-more">{tr("{0} more tasks validated", inspection.entries.length - 6)}</small> : null}
              {inspection.warnings.map((warning) => <Notice key={warning} tone="warning"><strong>{tr("Check notices")}</strong><p>{warning}</p></Notice>)}
              {hardBlockedReasons.map((reason) => <Notice key={reason} tone="danger"><strong>{tr("Hard environment blocks · cannot be bypassed")}</strong><p>{reason}</p></Notice>)}
              {overrideableReasons.map((reason) => <Notice key={reason} tone="warning"><strong>{tr("Resource limits · can be overridden explicitly")}</strong><p>{reason}</p></Notice>)}
              {inspection.admissible ? <Notice tone="success"><CheckCircle2 size={15} /><strong>{tr("This configuration can be queued")}</strong></Notice> : form.admission_override ? <Notice tone="warning"><AlertTriangle size={15} /><strong>{tr("Will be queued with an admission override")}</strong></Notice> : null}
            </>
          ) : !inspectError && !localSourceError ? <div className="inspection-placeholder"><span /><span /><span /></div> : null}
        </aside>
      </form>
    </Dialog>
  );
}
