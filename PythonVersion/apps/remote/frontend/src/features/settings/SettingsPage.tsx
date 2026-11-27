import { useCallback, useEffect, useState, type FormEvent } from "react";
import { ChevronDown, KeyRound, Save, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { api } from "../../api/client";
import { Badge, Button, EmptyState, LoadingBlock, PageHeader } from "../../components/ui";
import { usePolling } from "../../hooks/usePolling";
import { formatDateTime } from "../../lib/format";
import type { RemoteSettings } from "../../types";
import { tr } from "../../lib/i18n";

export function SettingsPage({ pollSeconds, refreshNonce, onPollSecondsChange, onResourcePollSecondsChange }: { pollSeconds: number; refreshNonce: number; onPollSecondsChange: (seconds: number) => void; onResourcePollSecondsChange: (seconds: number) => void }) {
  const loader = useCallback(async () => {
    const [settings, dashboard] = await Promise.all([api.settings.get(), api.dashboard()]);
    return { settings, audit: dashboard.audit_events };
  }, [refreshNonce]);
  const { data, loading, error, refresh } = usePolling(loader, Math.max(pollSeconds, 10));
  const [form, setForm] = useState<RemoteSettings | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (data?.settings && (!form || !dirty)) {
      setForm({
        ...data.settings,
        verify_artifact_checksums: true,
        delete_remote_results_after_transfer: true,
      });
    }
  }, [data?.settings, dirty, form]);

  const update = <K extends keyof RemoteSettings>(key: K, value: RemoteSettings[K]) => {
    setForm((current) => current ? { ...current, [key]: value } : current);
    setDirty(true);
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!form) return;
    setSaving(true);
    try {
      const saved = await api.settings.save({
        ...form,
        verify_artifact_checksums: true,
        delete_remote_results_after_transfer: true,
      });
      setForm({
        ...saved,
        verify_artifact_checksums: true,
        delete_remote_results_after_transfer: true,
      });
      setDirty(false);
      onPollSecondsChange(saved.default_poll_interval_seconds);
      onResourcePollSecondsChange(saved.resource_sampling_interval_seconds);
      toast.success(tr("Settings saved"), { description: tr("The resource sampling period applies to background monitoring at once; the run progress polling interval becomes the default for later runs.") });
      await refresh();
    } catch (saveError) { toast.error(tr("Save failed"), { description: saveError instanceof Error ? saveError.message : tr("Unknown error") }); }
    finally { setSaving(false); }
  };

  if (loading && !data) return <LoadingBlock label={tr("Reading the security policy and audit records…")} />;
  if (!data || !form) return <EmptyState title={tr("Cannot read the settings")} description={error?.message ?? tr("Check that the local service is running.")} />;

  return (
    <div className="page">
      <PageHeader title={tr("Settings and audit")} description={tr("Manage the local default run parameters, sampling periods and the audit of operations")} actions={<Button tone="primary" type="submit" form="settings-form" disabled={!dirty || saving} icon={<Save size={15} />}>{saving ? tr("Saving…") : dirty ? tr("Save changes") : tr("Saved")}</Button>} />

      <div className="settings-layout">
        <form id="settings-form" className="panel settings-form" onSubmit={(event) => void submit(event)}>
          <header className="section-title"><h2>{tr("Connection and run defaults")}</h2><p>{tr("Local controller paths and the defaults of polling, parallelism and resource sampling.")}</p></header>
          <div className="form-grid">
            <label className="field field--wide"><span>{tr("Local workspace")}</span><input value={form.workspace_path} readOnly /><small>{tr("Read-only. Code snapshots are created from this workspace.")}</small></label>
            <label className="field field--wide"><span>{tr("Default SSH private key")}</span><div className="input-with-icon"><KeyRound size={15} /><input required value={form.default_ssh_key_path} onChange={(event) => update("default_ssh_key_path", event.target.value)} /></div></label>
            <label className="field field--wide"><span>{tr("Default remote root folder")}</span><input required value={form.default_remote_root} onChange={(event) => update("default_remote_root", event.target.value)} /></label>
            <label className="field"><span>{tr("Run progress polling interval (s)")}</span><input min={1} max={300} type="number" value={form.default_poll_interval_seconds} onChange={(event) => update("default_poll_interval_seconds", Number(event.target.value))} /><small>{tr("The default interval for reading progress and resource events of new runs.")}</small></label>
            <label className="field"><span>{tr("Default parallel runs")}</span><input min={1} max={256} type="number" value={form.default_max_parallel_runs} onChange={(event) => update("default_max_parallel_runs", Number(event.target.value))} /></label>
            <label className="field"><span>{tr("Probe timeout (s)")}</span><input min={3} max={300} type="number" value={form.probe_timeout_seconds} onChange={(event) => update("probe_timeout_seconds", Number(event.target.value))} /></label>
            <label className="field"><span>{tr("Resource sampling period (s)")}</span><input min={1} max={300} type="number" value={form.resource_sampling_interval_seconds} onChange={(event) => update("resource_sampling_interval_seconds", Number(event.target.value))} /><small>{tr("The background controller samples at this period; the server page and the overview only read the local database.")}</small></label>
            <label className="field"><span>{tr("Resource history retention (h)")}</span><input min={1} max={720} type="number" value={form.resource_history_retention_hours} onChange={(event) => update("resource_history_retention_hours", Number(event.target.value))} /><small>{tr("72 hours by default; expired resource samples are removed in the background.")}</small></label>
          </div>
          <div className="policy-checks">
            <label className="setting-toggle-row">
              <span>
                <strong>{tr("Require a clean Git working tree")}</strong>
                <small>{tr("When off, uncommitted changes are still included in the immutable snapshot and recorded as dirty.")}</small>
              </span>
              <input
                type="checkbox"
                role="switch"
                className="apple-switch"
                checked={form.require_clean_git}
                onChange={(event) => update("require_clean_git", event.target.checked)}
              />
            </label>
            <label className="setting-toggle-row setting-toggle-row--locked">
              <span>
                <strong>{tr("Verify the checksums of collected files")}</strong>
                <small>{tr("A fixed invariant of result integrity; always on.")}</small>
              </span>
              <input type="checkbox" role="switch" className="apple-switch" checked disabled readOnly />
            </label>
            <label className="setting-toggle-row setting-toggle-row--locked">
              <span>
                <strong>{tr("Clean up results on the executing machine after validation")}</strong>
                <small>{tr("Done only after the declared artifacts have arrived completely on this computer and passed validation; always on.")}</small>
              </span>
              <input type="checkbox" role="switch" className="apple-switch" checked disabled readOnly />
            </label>
          </div>
          <details className="policy-disclosure">
            <summary><ShieldCheck size={14} /><span>{tr("Security and reproducibility rules that cannot be overridden (SSH private key · frozen snapshot · separate storage · cleanup after validation)")}</span><ChevronDown size={14} /></summary>
            <ul>
              <li><strong>{tr("SSH private keys on this computer only")}</strong>{tr(": no passwords are stored and private keys are never copied to the remote machine; remote key authentication cannot be skipped by an admission override.")}</li>
              <li><strong>{tr("Execution from a frozen snapshot")}</strong>{tr(": the commit, dirty state, snapshot SHA and task SHA are archived together, and execution starts once they match.")}</li>
              <li><strong>{tr("Separate archived storage")}</strong>{tr(": every run keeps its own artifacts, exit code and error context.")}</li>
              <li><strong>{tr("Cleanup after validation")}</strong>{tr(": the temporary folder on the executing machine owned by the marker is deleted only after local validation passes.")}</li>
            </ul>
          </details>
        </form>
      </div>

      <section className="panel audit-panel">
        <header className="panel__header"><h2>{tr("Recent audit events")}</h2><span className="panel-count">{tr("{0} entries", data.audit.length)}</span></header>
        <div className="audit-table-wrap"><table className="audit-table"><thead><tr><th>{tr("Time")}</th><th>{tr("Action")}</th><th>{tr("Target")}</th><th>{tr("Description")}</th><th>{tr("Result")}</th></tr></thead><tbody>{data.audit.map((event) => <tr key={event.id}><td><time>{formatDateTime(event.timestamp)}</time><small>{event.actor}</small></td><td><code>{event.action}</code></td><td>{event.target}</td><td>{event.detail}</td><td><Badge tone={event.outcome === "success" ? "success" : event.outcome === "warning" ? "warning" : "danger"}>{event.outcome === "success" ? tr("Success") : event.outcome === "warning" ? tr("Notice") : tr("Failed")}</Badge></td></tr>)}</tbody></table></div>
        <div className="audit-card-list">{data.audit.map((event) => <article key={event.id}><header><code>{event.action}</code><Badge tone={event.outcome === "success" ? "success" : event.outcome === "warning" ? "warning" : "danger"}>{event.outcome === "success" ? tr("Success") : event.outcome === "warning" ? tr("Notice") : tr("Failed")}</Badge></header><strong>{event.target}</strong><p>{event.detail}</p><small>{formatDateTime(event.timestamp)} · {event.actor}</small></article>)}</div>
      </section>
    </div>
  );
}
