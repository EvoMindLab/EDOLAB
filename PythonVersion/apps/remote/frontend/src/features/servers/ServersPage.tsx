import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { AlertTriangle, Check, ChevronDown, CircleAlert, Cpu, Edit3, Fingerprint, HardDrive, KeyRound, MemoryStick, Plus, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "../../api/client";
import { createPolling } from "../../lib/polling";
import { Badge, Button, ConfirmDialog, Dialog, EmptyState, IconButton, LoadingBlock, PageHeader, ServerStatusPill, Tabs, type BadgeTone } from "../../components/ui";
import { usePolling } from "../../hooks/usePolling";
import { formatRelativeTime } from "../../lib/format";
import type { RemoteServer, ServerCheckStatus, ServerEnvironmentCheck, ServerInput, ServerResourceHistoryResponse, ServerResourceMonitor, ServerSystemInfo } from "../../types";
import { ResourceFreshnessBadge, ResourceMiniSummary, ServerResourceMonitor as ServerResourceMonitorPanel, resourceMonitorFromApi } from "./ServerResourceMonitor";
import { tr } from "../../lib/i18n";

const emptyServer = (sshKeyPath: string, remoteRoot: string): ServerInput => ({
  name: "",
  host: "",
  port: 22,
  username: "",
  ssh_key_path: sshKeyPath,
  remote_root: remoteRoot,
  conda_env: "EDOLAB",
  enabled: true,
});

const finiteNumber = (value: number | null | undefined, fallback = 0): number => Number.isFinite(value) ? Number(value) : fallback;
const checkStatusLabel: Record<ServerCheckStatus, string> = { passed: tr("Passed"), warning: tr("Notice"), failed: tr("Failed"), unknown: tr("Unconfirmed") };

const containsIssue = (system: ServerSystemInfo, terms: string[]): string | undefined => system.environment_issues?.find((issue) => {
  const normalized = issue.toLowerCase();
  return terms.some((term) => normalized.includes(term));
});

const legacyChecks = (server: RemoteServer): ServerEnvironmentCheck[] => {
  const system = server.system;
  const targetLabel = server.target_kind === "local" ? tr("This computer") : tr("Remote");
  if (!system) {
    return [{
      key: "environment_probe",
      label: tr("Environment probe"),
      status: server.probe_error ? "failed" : "unknown",
      detail: server.probe_error ?? tr("The environment check has not been completed."),
      blocking: true,
    }];
  }

  const condaIssue = containsIssue(system, ["conda environment"]);
  const pythonIssue = containsIssue(system, ["python 3.11", "python version"]);
  const pipIssue = containsIssue(system, ["pip"]);
  const workspaceIssue = containsIssue(system, ["remote root", "workspace", "not writable"]);
  const diskIssue = containsIssue(system, ["disk", "free space"]);
  const loadIssue = containsIssue(system, ["load"]);
  const pythonVersion = system.python_version?.trim();
  const pythonIs311 = Boolean(pythonVersion && /^3\.11(?:\.|$)/.test(pythonVersion));

  return [
    {
      key: "conda_environment",
      label: tr("Conda environment {0}", server.conda_env),
      status: system.conda_environment_available === true ? "passed" : system.conda_environment_available === false ? "failed" : "unknown",
      detail: condaIssue ?? tr("{0}: {1}", system.conda_env ?? server.conda_env, system.conda_environment_available === true ? tr("available") : system.conda_environment_available === false ? tr("unavailable") : tr("not confirmed yet")),
      blocking: true,
    },
    {
      key: "environment_python",
      label: "Python 3.11",
      status: pythonVersion ? (pythonIs311 ? "passed" : "failed") : "unknown",
      detail: pythonIssue ?? (pythonVersion ? tr("{0} reports Python {1}", targetLabel, pythonVersion) : tr("Could not confirm the Python version of the EDOLAB environment.")),
      blocking: true,
    },
    {
      key: "environment_pip",
      label: tr("pip in the environment"),
      status: pipIssue ? "failed" : system.environment_eligible === true ? "passed" : "unknown",
      detail: pipIssue ?? (system.environment_eligible === true ? tr("pip is available.") : tr("The older probe snapshot did not report the pip state separately.")),
      blocking: true,
    },
    {
      key: "remote_workspace",
      label: tr("Working folder ({0})", targetLabel),
      status: system.disk_writable === true ? "passed" : system.disk_writable === false ? "failed" : "unknown",
      detail: workspaceIssue ?? tr("{0}: {1}", server.remote_root, system.disk_writable === true ? tr("writable") : system.disk_writable === false ? tr("not writable") : tr("not confirmed yet")),
      blocking: true,
    },
    {
      key: "remote_disk_space",
      label: tr("Free disk ({0})", targetLabel),
      status: diskIssue ? "failed" : Number.isFinite(system.disk_free_gb) ? "passed" : "unknown",
      detail: diskIssue ?? (Number.isFinite(system.disk_free_gb) ? tr("{0} GB free", system.disk_free_gb.toFixed(1)) : tr("Could not confirm the free space ({0}).", targetLabel)),
      blocking: true,
    },
    {
      key: "server_load",
      label: tr("Machine load"),
      status: loadIssue ? "warning" : system.load_ratio == null ? "unknown" : "passed",
      detail: loadIssue ?? (system.load_ratio == null ? tr("Could not obtain the load ratio.") : `load ratio ${system.load_ratio.toFixed(2)}`),
      blocking: false,
    },
    {
      key: "gpu_availability",
      label: "NVIDIA GPU",
      status: system.gpu_usable === true ? "passed" : system.gpu_usable === false ? "warning" : "unknown",
      detail: system.gpu_usable === true ? tr("{0} usable GPUs detected.", system.gpus.length) : system.gpu_error ?? tr("GPU not confirmed; only GPU tasks are checked separately at submission."),
      blocking: false,
    },
  ];
};

const loadThresholdFrom = (detail: string): number => {
  const match = detail.match(/(?:threshold|阈值)[^0-9]*([0-9]+(?:\.[0-9]+)?)/i);
  return match ? Number(match[1]) : 0.9;
};

const environmentChecks = (server: RemoteServer, monitor?: ServerResourceMonitor | null): ServerEnvironmentCheck[] => {
  const live = monitor?.state === "fresh" ? monitor.latest_success : undefined;
  const source = server.system?.checks?.length ? server.system.checks : legacyChecks(server);
  let sawLoadCheck = false;
  const normalized = source.map((rawCheck, index) => {
    const key = rawCheck.key || `check_${index}`;
    const label = rawCheck.label || key;
    const isLoadCheck = /load|负载/i.test(`${key} ${label}`);
    if (!isLoadCheck) {
      return {
        ...rawCheck,
        key,
        label,
        blocking: rawCheck.blocking ?? rawCheck.status === "failed",
      };
    }

    sawLoadCheck = true;
    const threshold = loadThresholdFrom(rawCheck.detail);
    const liveRatio = live?.load_ratio;
    return {
      ...rawCheck,
      key,
      label,
      status: liveRatio == null ? (rawCheck.status === "failed" ? "warning" : rawCheck.status) : liveRatio > threshold ? "warning" : "passed",
      detail: liveRatio == null ? rawCheck.detail : tr("Live load ratio {0}; warning threshold {1}. High load only warns and does not block submission.", liveRatio.toFixed(2), threshold.toFixed(2)),
      blocking: false,
    };
  });

  if (!sawLoadCheck && live?.load_ratio != null) {
    normalized.push({
      key: "server_load",
      label: tr("Machine load"),
      status: live.load_ratio > 0.9 ? "warning" : "passed",
      detail: tr("Live load ratio {0}; warning threshold 0.90. High load only warns and does not block submission.", live.load_ratio.toFixed(2)),
      blocking: false,
    });
  }
  return normalized;
};

const environmentSummary = (checks: ServerEnvironmentCheck[]): { tone: BadgeTone; label: string } => {
  const blockingFailures = checks.filter((check) => check.blocking && check.status === "failed").length;
  const warnings = checks.filter((check) => check.status === "warning").length;
  const unknown = checks.filter((check) => check.status === "unknown").length;
  if (blockingFailures) return { tone: "danger", label: tr("{0} required conditions failed", blockingFailures) };
  if (warnings) return { tone: "warning", label: tr("Required conditions passed · {0} notices", warnings) };
  if (unknown) return { tone: "neutral", label: tr("{0} items not confirmed yet", unknown) };
  return { tone: "success", label: tr("All checked conditions passed") };
};

const statusIcon = (status: ServerCheckStatus) => status === "passed" ? <Check size={14} /> : status === "warning" ? <AlertTriangle size={14} /> : <CircleAlert size={14} />;

function ServerResourceSection({ serverId, summary, pollSeconds }: { serverId: string; summary?: ServerResourceMonitor | null; pollSeconds: number }) {
  const [windowMinutes, setWindowMinutes] = useState(20);
  const [history, setHistory] = useState<ServerResourceHistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<Error | null>(null);
  const afterId = useRef<number | null>(null);

  useEffect(() => {
    let active = true;
    const interval = Math.max(summary?.sample_interval_seconds ?? pollSeconds, 1);
    const retainedPoints = Math.min(10_000, Math.max(300, Math.ceil(windowMinutes * 60 / interval) + 20));
    afterId.current = null;
    setHistory(null);
    setHistoryError(null);

    const read = async (initial: boolean) => {
      if (!active) return;
      try {
        const response = initial || afterId.current == null
          ? await api.servers.resourceHistory(serverId, { since: new Date(Date.now() - windowMinutes * 60_000).toISOString(), limit: retainedPoints })
          : await api.servers.resourceHistory(serverId, { afterId: afterId.current ?? undefined, limit: Math.min(1_000, retainedPoints) });
        if (!active) return;
        const newestId = response.samples.reduce((maximum, sample) => Math.max(maximum, sample.id), afterId.current ?? 0);
        // Only advance to an ID that the history response actually returned.
        // The monitor summary is serialized after the history query and may
        // observe a newer concurrent sample; jumping to it would skip data.
        afterId.current = response.next_after_id ?? (newestId > 0 ? newestId : afterId.current);
        setHistory((current) => {
          if (initial || current == null) return response;
          const cutoff = Date.now() - windowMinutes * 60_000;
          const merged = new Map(current.samples.map((sample) => [sample.id, sample]));
          response.samples.forEach((sample) => merged.set(sample.id, sample));
          const samples = [...merged.values()]
            .filter((sample) => new Date(sample.sampled_at).getTime() >= cutoff)
            .sort((left, right) => left.id - right.id)
            .slice(-retainedPoints);
          return { ...response, samples, next_after_id: response.next_after_id };
        });
        setHistoryError(null);
      } catch (error) {
        if (active) setHistoryError(error instanceof Error ? error : new Error(tr("Could not read the local resource history")));
      }
    };

    const polling = createPolling(() => read(afterId.current == null), () => interval * 1000);
    return () => {
      active = false;
      polling.dispose();
    };
  }, [pollSeconds, serverId, summary?.sample_interval_seconds, windowMinutes]);

  const monitor = history?.resource_monitor ?? summary;
  const view = resourceMonitorFromApi(monitor, history?.samples ?? []);

  return (
    <div className="server-resource-section">
      <div className="resource-window-control" aria-label={tr("Resource history time range")}>
        <span>{tr("History range")}</span><small>{tr("The last hour keeps the raw samples; earlier data is aggregated per minute, with the mean for CPU/memory and the last sample of the minute for GPUs.")}</small>
        <Tabs
          variant="segmented"
          ariaLabel={tr("Resource history time range")}
          value={String(windowMinutes)}
          onChange={(next) => setWindowMinutes(Number(next))}
          items={[10, 20, 60].map((minutes) => ({ value: String(minutes), label: tr("{0} min", minutes) }))}
        />
      </div>
      {historyError ? <div className="resource-read-error"><AlertTriangle size={15} /><div><strong>{tr("Could not read the local history")}</strong><p>{tr("{0}; the summary still uses the latest state in the database.", historyError.message)}</p></div></div> : null}
      <ServerResourceMonitorPanel monitor={view} />
    </div>
  );
}

function ServerEditor({ open, server, defaultSshKey, defaultRemoteRoot, busy, onClose, onSave }: { open: boolean; server: RemoteServer | null; defaultSshKey: string; defaultRemoteRoot: string; busy: boolean; onClose: () => void; onSave: (input: ServerInput) => void }) {
  const [form, setForm] = useState<ServerInput>(() => emptyServer(defaultSshKey, defaultRemoteRoot));
  const editingLocal = server?.target_kind === "local";

  useEffect(() => {
    if (!open) return;
    setForm(server ? {
      name: server.name, host: server.host, port: server.port, username: server.username,
      ssh_key_path: server.ssh_key_path, remote_root: server.remote_root,
      conda_env: server.conda_env, enabled: server.enabled,
    } : emptyServer(defaultSshKey, defaultRemoteRoot));
  }, [defaultRemoteRoot, defaultSshKey, open, server]);

  const update = <K extends keyof ServerInput>(key: K, value: ServerInput[K]) => setForm((current) => ({ ...current, [key]: value }));
  const submit = (event: FormEvent) => { event.preventDefault(); onSave(form); };

  return (
    <Dialog
      open={open}
      title={editingLocal ? tr("Edit local scheduling") : server ? tr("Edit SSH server") : tr("Add SSH server")}
      description={editingLocal ? tr("The local execution identity, EDOLAB environment and isolated folders are managed by the platform; here you can change the name and whether new tasks are allowed.") : tr("Only SSH private-key authentication on this computer is used; the private key is never uploaded or written to the remote machine.")}
      onClose={onClose}
      footer={<><Button onClick={onClose} disabled={busy}>{tr("Cancel")}</Button><Button tone="primary" type="submit" form="server-form" disabled={busy}>{busy ? tr("Saving…") : tr("Save server")}</Button></>}
    >
      <form id="server-form" className="form-grid" onSubmit={submit}>
        <label className="field field--wide"><span>{tr("Display name")}</span><input required value={form.name} onChange={(event) => update("name", event.target.value)} placeholder={tr("For example: lab A6000")} /></label>
        {!editingLocal ? <>
          <label className="field"><span>{tr("Host / IP")}</span><input required value={form.host} onChange={(event) => update("host", event.target.value)} placeholder="compute.example.org" spellCheck={false} /></label>
          <label className="field field--small"><span>{tr("SSH port")}</span><input required min={1} max={65535} type="number" value={form.port} onChange={(event) => update("port", Number(event.target.value))} /></label>
          <label className="field"><span>{tr("User name")}</span><input required value={form.username} onChange={(event) => update("username", event.target.value)} placeholder="researcher" spellCheck={false} /></label>
          <label className="field field--wide"><span>{tr("Private key path on this computer")}</span><div className="input-with-icon"><KeyRound size={15} /><input required value={form.ssh_key_path} onChange={(event) => update("ssh_key_path", event.target.value)} placeholder="~/.ssh/id_ed25519" spellCheck={false} /></div><small>{tr("The backend only reads a local path and accepts no password field.")}</small></label>
          <label className="field field--wide"><span>{tr("Remote working root folder")}</span><input required value={form.remote_root} onChange={(event) => update("remote_root", event.target.value)} placeholder="~/edolab-remote" spellCheck={false} /><small>{tr("Code snapshots, run workspaces and temporary results are all kept in this folder.")}</small></label>
          <label className="field field--wide"><span>{tr("Conda environment name")}</span><input required maxLength={128} pattern="[A-Za-z0-9_.-]+" value={form.conda_env} onChange={(event) => update("conda_env", event.target.value)} placeholder="EDOLAB" spellCheck={false} /><small>{tr("Remote tasks activate this environment; after changing the name, run the environment check again.")}</small></label>
        </> : <div className="selected-server-trust field--wide"><strong>{tr("Isolated local execution")}</strong><code>{form.remote_root}</code><small>{tr("Every run uses a frozen copy of the code; the shared editable EDOLAB installation is never pointed at a temporary folder.")}</small></div>}
        <label className="check-field field--wide"><input type="checkbox" checked={form.enabled} onChange={(event) => update("enabled", event.target.checked)} /><span><strong>{tr("Allow scheduling on this machine")}</strong><small>{tr("When off, background sampling continues and the history is kept, but new runs cannot choose it.")}</small></span></label>
      </form>
    </Dialog>
  );
}

export function ServersPage({ pollSeconds, refreshNonce }: { pollSeconds: number; refreshNonce: number }) {
  const loader = useCallback(async () => {
    const [servers, settings] = await Promise.all([api.servers.list(), api.settings.get()]);
    return { servers, settings };
  }, [refreshNonce]);
  const { data, loading, error, refresh } = usePolling(loader, Math.max(pollSeconds, 1));
  const [editorOpen, setEditorOpen] = useState(false);
  const [editing, setEditing] = useState<RemoteServer | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<RemoteServer | null>(null);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [probingId, setProbingId] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());

  const openNew = () => { setEditing(null); setEditorOpen(true); };
  const openEdit = (server: RemoteServer) => { setEditing(server); setEditorOpen(true); };
  const toggleExpanded = (serverId: string) => setExpandedIds((current) => {
    const next = new Set(current);
    if (next.has(serverId)) next.delete(serverId);
    else next.add(serverId);
    return next;
  });

  const saveServer = async (input: ServerInput) => {
    setSaving(true);
    try {
      if (editing) await api.servers.update(editing.id, input);
      else await api.servers.create(input);
      toast.success(editing?.target_kind === "local" ? tr("Local scheduling settings updated") : editing ? tr("Server settings updated") : tr("Server added"), { description: editing?.target_kind === "local" ? tr("Turning scheduling off does not delete the history or stop resource sampling.") : tr("Run the environment check now; the first connection pins the host key with TOFU.") });
      setEditorOpen(false);
      await refresh();
    } catch (saveError) {
      toast.error(tr("Save failed"), { description: saveError instanceof Error ? saveError.message : tr("Unknown error") });
    } finally { setSaving(false); }
  };

  const probe = async (server: RemoteServer) => {
    setProbingId(server.id);
    try {
      await api.servers.probe(server.id);
      toast.success(server.name + tr(": environment check completed"), { description: server.target_kind === "local" ? tr("Review the local EDOLAB, folder and resource conditions one by one; high load only warns and does not block submission.") : tr("Review the check results one by one and verify the full SSH fingerprint; high load only warns and does not block submission.") });
      setExpandedIds((current) => new Set(current).add(server.id));
      await refresh();
    } catch (probeError) {
      toast.error(tr("Environment probe failed"), { description: probeError instanceof Error ? probeError.message : tr("Unknown error") });
    } finally { setProbingId(null); }
  };

  const remove = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await api.servers.remove(deleteTarget.id);
      toast.success(tr("Server deleted"));
      setDeleteTarget(null);
      await refresh();
    } catch (removeError) {
      toast.error(tr("Deletion failed"), { description: removeError instanceof Error ? removeError.message : tr("Unknown error") });
    } finally { setDeleting(false); }
  };

  if (loading && !data) return <LoadingBlock label={tr("Reading the server settings…")} />;
  if (!data) return <EmptyState title={tr("Cannot read the servers")} description={error?.message ?? tr("Check that the local service is running.")} action={<Button onClick={openNew}>{tr("Add a server anyway")}</Button>} />;

  return (
    <div className="page">
      <PageHeader title={tr("Machines")} description={tr("This computer and SSH machines (local resource samples read every {0} s)", Math.max(pollSeconds, 1))} actions={<Button tone="primary" onClick={openNew} icon={<Plus size={15} />}>{tr("Add SSH server")}</Button>} />

      <section className="server-grid" aria-label={tr("Machine list")}>
        {data.servers.map((server) => {
          const monitor = server.resource_monitor;
          const live = monitor?.latest_success;
          const system = server.system;
          const memoryTotalGb = finiteNumber(live?.memory_total_gb, finiteNumber(system?.memory_total_gb));
          const checks = environmentChecks(server, monitor);
          const summary = environmentSummary(checks);
          const expanded = expandedIds.has(server.id);
          const detailsId = `server-details-${server.id}`;
          const hasResourceSnapshot = Boolean(live || system);
          const monitorView = resourceMonitorFromApi(monitor);

          return (
            <article className={("server-card" + (!server.enabled ? " is-disabled" : "") + (expanded ? " is-expanded" : "")).trim()} key={server.id}>
              <header className="server-card__summary">
                <button className="server-card__toggle" type="button" aria-label={tr("{0} details of {1}", expanded ? tr("Collapse") : tr("Expand"), server.name)} aria-expanded={expanded} aria-controls={detailsId} onClick={() => toggleExpanded(server.id)}>
                  <span className="server-card__identity">
                    <span className="server-card__title"><strong>{server.name}</strong>{!server.enabled ? <Badge tone="neutral">{tr("Disabled")}</Badge> : null}</span>
                    <small>{server.target_kind === "local" ? tr("Local executor") : `${server.username}@${server.host}:${server.port}`}</small>
                  </span>
                  <ResourceMiniSummary monitor={monitorView} />
                  <span className="server-card__summary-state">
                    <ServerStatusPill status={probingId === server.id ? "probing" : server.status} />
                    {monitorView.state === "stale" || monitorView.state === "failed" ? <ResourceFreshnessBadge state={monitorView.state} /> : null}
                    <span className={"server-check-inline server-check-inline--" + summary.tone} title={summary.label}>{summary.label}</span>
                    <ChevronDown size={16} aria-hidden="true" />
                  </span>
                </button>
                <div className="server-card__quick-actions">
                  <Button onClick={() => void probe(server)} disabled={probingId === server.id} icon={<RefreshCw size={14} className={probingId === server.id ? "is-spinning" : ""} />}>{probingId === server.id ? tr("Checking") : tr("Environment check")}</Button>
                  <IconButton aria-label={tr("Edit ") + server.name} title={tr("Edit")} onClick={() => openEdit(server)}><Edit3 size={16} /></IconButton>
                  {!server.system_managed ? <IconButton className="is-danger" aria-label={tr("Delete ") + server.name} title={tr("Delete")} onClick={() => setDeleteTarget(server)}><Trash2 size={16} /></IconButton> : null}
                </div>
              </header>

              {expanded ? (
                <div className="server-card__details" id={detailsId}>
                  <section className="server-detail-section server-detail-section--connection" aria-labelledby={`${detailsId}-connection`}>
                    <header><h3 id={`${detailsId}-connection`}>{tr("Connection and identity")}</h3><small>{tr("Last probe: {0}", formatRelativeTime(server.last_probe_at))}</small></header>
                    <div className="server-card__connection">
                      {server.target_kind === "ssh" ? <span><KeyRound size={14} /><code>{server.ssh_key_path}</code></span> : null}
                      <span><HardDrive size={14} /><code>{server.remote_root}</code></span>
                      <span><Cpu size={14} /><code>Conda {server.conda_env}</code></span>
                    </div>
                    <div className={("transport-status " + (server.status === "online" ? "is-reachable" : server.status === "error" ? "is-error" : "is-unknown")).trim()}>
                      <strong>{server.target_kind === "local" ? (server.status === "online" ? tr("Local executor available") : tr("Local environment not ready")) : server.status === "online" ? tr("Reachable with SSH key authentication") : server.status === "error" ? tr("SSH or check error") : tr("SSH reachability not confirmed yet")}</strong>
                      <span>{server.status === "online" ? tr("The execution channel and the environment conditions are judged separately.") : server.probe_error ?? tr("The execution conditions are confirmed after one environment check.")}</span>
                    </div>
                    {server.target_kind === "ssh" ? <div className={("fingerprint-panel " + (server.fingerprint ? "is-pinned" : "is-missing")).trim()}>
                      <span><Fingerprint size={16} /></span>
                      <div><strong>{tr("SSH host key fingerprint")}</strong><code>{server.fingerprint ?? tr("Not pinned yet; the environment check must be completed first")}</code><p>{server.fingerprint ? tr("Verify the full fingerprint through a trusted channel before the first run; later changes of the host key are refused.") : tr("The first environment check pins the host key with OpenSSH accept-new (TOFU).")}</p></div>
                    </div> : <div className="fingerprint-panel is-pinned"><span><Fingerprint size={16} /></span><div><strong>{tr("Local identity fingerprint")}</strong><code>{server.fingerprint ?? tr("Created after the environment check")}</code><p>{tr("Identifies whether the database was moved to another machine; local execution does not read SSH private keys.")}</p></div></div>}
                  </section>

                  <section className="server-detail-section" aria-labelledby={`${detailsId}-resources`}>
                    <header>
                      <h3 id={`${detailsId}-resources`}>{tr("Resource monitoring")}</h3>
                      <small>{tr("Local samples refreshed every {0} s", Math.max(monitor?.sample_interval_seconds ?? pollSeconds, 1))}</small>
                    </header>
                    {hasResourceSnapshot ? (
                      <div className="system-facts">
                        <span><Cpu size={15} /><small>CPU</small><strong>{live?.cpu_physical_cores != null || system ? `${live?.cpu_physical_cores ?? system?.cpu_physical ?? "—"}P / ${live?.cpu_logical_cores ?? system?.cpu_logical ?? "—"}T` : tr("Unconfirmed")}</strong></span>
                        <span><MemoryStick size={15} /><small>{tr("Memory")}</small><strong>{memoryTotalGb ? `${memoryTotalGb.toFixed(1)} GB` : tr("Unconfirmed")}</strong></span>
                        <span><HardDrive size={15} /><small>{tr("Free disk")}</small><strong>{system ? `${system.disk_free_gb.toFixed(1)} GB` : tr("Unconfirmed")}</strong></span>
                      </div>
                    ) : null}
                    <ServerResourceSection serverId={server.id} summary={monitor} pollSeconds={pollSeconds} />
                  </section>

                  <section className="server-detail-section" aria-labelledby={`${detailsId}-checks`}>
                    <header><h3 id={`${detailsId}-checks`}>{tr("Environment checks")}</h3><Badge tone={summary.tone}>{summary.label}</Badge></header>
                    <div className="server-check-list">
                      {checks.map((check) => (
                        <div className={"server-check server-check--" + check.status} key={check.key}>
                          <span className="server-check__icon">{statusIcon(check.status)}</span>
                          <div className="server-check__name"><strong>{tr(check.label)}</strong><span>{check.blocking ? tr("Required conditions") : tr("Non-blocking notices")}</span></div>
                          <p>{check.detail}</p>
                          <span className="server-check__status">{checkStatusLabel[check.status]}</span>
                        </div>
                      ))}
                    </div>
                    {system ? <div className="environment-line"><span>{system.hostname}</span><span>{system.os}</span><span>Python {system.python_version ?? tr("Unconfirmed")}</span><span>Conda {system.conda_env ?? server.conda_env}</span></div> : null}
                  </section>

                  <footer className="server-card__footer"><Button tone="quiet" onClick={() => toggleExpanded(server.id)} icon={<ChevronDown size={14} className="is-rotated" />}>{tr("Collapse details")}</Button></footer>
                </div>
              ) : null}
            </article>
          );
        })}
        {!data.servers.length ? <EmptyState title={tr("No servers yet")} description={tr("Add a remote server that can be reached with an SSH private key on this computer.")} action={<Button tone="primary" onClick={openNew}>{tr("Add server")}</Button>} /> : null}
      </section>

      <ServerEditor open={editorOpen} server={editing} defaultSshKey={data.settings.default_ssh_key_path} defaultRemoteRoot={data.settings.default_remote_root} busy={saving} onClose={() => setEditorOpen(false)} onSave={(input) => void saveServer(input)} />
      <ConfirmDialog open={Boolean(deleteTarget)} title={tr("Delete the server settings?")} message={tr("The connection settings of “") + (deleteTarget?.name ?? "") + tr("” will be deleted, but past run records and local results are not. While any run history refers to the server, the service refuses the deletion so that the evidence stays intact.")} confirmLabel={tr("Delete server")} destructive busy={deleting} onClose={() => setDeleteTarget(null)} onConfirm={() => void remove()} />
    </div>
  );
}
