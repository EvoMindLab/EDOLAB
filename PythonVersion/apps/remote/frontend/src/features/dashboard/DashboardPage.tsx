import { useCallback } from "react";
import { ArrowRight, Plus } from "lucide-react";
import { api } from "../../api/client";
import { usePolling } from "../../hooks/usePolling";
import { formatBytes, formatRelativeTime, isTerminalRun } from "../../lib/format";
import type { RemoteRun, ViewId } from "../../types";
import { Button, EmptyState, LoadingBlock, PageHeader, RunStatusPill, ServerStatusPill } from "../../components/ui";
import { ResourceFreshnessBadge, ResourceMiniSummary, resourceMonitorFromApi } from "../servers/ServerResourceMonitor";
import { tr } from "../../lib/i18n";

export function DashboardPage({ pollSeconds, refreshNonce, onNewRun, onViewChange, onOpenRun }: { pollSeconds: number; refreshNonce: number; onNewRun: () => void; onViewChange: (view: ViewId) => void; onOpenRun: (run: RemoteRun) => void }) {
  const loader = useCallback(async () => {
    return api.dashboard();
  }, [refreshNonce]);
  const { data, loading, error } = usePolling(loader, Math.max(pollSeconds, 1));

  if (loading && !data) return <LoadingBlock label={tr("Gathering the state of the experiments…")} />;
  if (!data) return <EmptyState title={tr("The overview cannot be read right now")} description={error?.message ?? tr("Check that the local service is running.")} />;

  const activeRuns = data.active_runs ?? data.recent_runs.filter((run) => !isTerminalRun(run.status));
  const onlineServers = data.servers.filter((serverItem) => serverItem.status === "online");

  return (
    <div className="page page--overview">
      <PageHeader
        title={tr("Overview")}
        description={tr("This computer, SSH machines and the experiments now running")}
        actions={<Button tone="primary" onClick={onNewRun} icon={<Plus size={15} />}>{tr("New run")}</Button>}
      />

      <section className="metrics-strip panel" aria-label={tr("Key figures")}>
        <button type="button" className="metrics-strip__item" onClick={() => onViewChange("servers")}>
          <span className="metrics-strip__label">{tr("Machines online")}</span>
          <strong>{onlineServers.length}<em> / {data.servers.length}</em></strong>
          <small>{data.servers.length ? tr("View machines and environments") : tr("Setting up this computer")}</small>
        </button>
        <button type="button" className="metrics-strip__item" onClick={() => onViewChange("runs")}>
          <span className="metrics-strip__label">{tr("Running")}</span>
          <strong className={data.active_run_count ? "is-accent" : ""}>{data.active_run_count}</strong>
          <small>{activeRuns.length ? activeRuns[0].current_message : tr("No active tasks")}</small>
        </button>
        <button type="button" className="metrics-strip__item" onClick={() => onViewChange("analysis")}>
          <span className="metrics-strip__label">{tr("Archived runs")}</span>
          <strong>{data.completed_run_count}</strong>
          <small>{tr("{0} in total", formatBytes(data.transferred_bytes))}</small>
        </button>
        <button type="button" className={"metrics-strip__item" + (data.failed_run_count ? " has-alert" : "")} onClick={() => onViewChange("runs")}>
          <span className="metrics-strip__label">{tr("Failed runs")}</span>
          <strong>{data.failed_run_count}</strong>
          <small>{data.failed_run_count ? tr("View exit codes and logs") : tr("No failures")}</small>
        </button>
      </section>

      <div className="overview-grid">
        <section className="panel panel--runs">
          <header className="panel__header"><h2>{tr("Current runs")}</h2><button className="text-link" onClick={() => onViewChange("runs")}>{tr("All records")}{" "}<ArrowRight size={14} /></button></header>
          {activeRuns.length ? (
            <div className="run-brief-list">
              {activeRuns.map((run) => (
                <button key={run.id} className="run-brief" onClick={() => onOpenRun(run)}>
                  <div className="run-brief__top"><RunStatusPill status={run.status} /><span>{run.server_name}</span><time>{formatRelativeTime(run.started_at ?? run.created_at)}</time></div>
                  <strong>{run.task_label ?? run.task_path}</strong>
                  <p>{run.current_message ?? tr("Waiting for a stage update")}</p>
                  <div className="progress-line"><span style={{ width: run.progress_percent + "%" }} /></div>
                  <small>{tr("{0} / {1} run units · {2}%", run.completed_units, run.total_units, Math.round(run.progress_percent))}</small>
                </button>
              ))}
            </div>
          ) : <EmptyState title={tr("No active runs")} description={tr("Choose a task and a machine to start a run.")} action={<Button onClick={onNewRun}>{tr("New run")}</Button>} />}
        </section>

        <section className="panel panel--servers">
          <header className="panel__header"><h2>{tr("Machine load")}</h2><button className="text-link" onClick={() => onViewChange("servers")}>{tr("Monitoring details")}{" "}<ArrowRight size={14} /></button></header>
          <div className="server-brief-list">
            {data.servers.slice(0, 4).map((serverItem) => {
              const monitor = resourceMonitorFromApi(serverItem.resource_monitor);
              return (
                <article className="server-brief" key={serverItem.id}>
                  <header>
                    <div>
                      <strong>{serverItem.name}</strong>
                      <small>{serverItem.target_kind === "local" ? tr("Local executor") : `${serverItem.username}@${serverItem.host}:${serverItem.port}`}</small>
                    </div>
                    <span className="server-brief__state">
                      <ServerStatusPill status={serverItem.status} />
                      {monitor.state !== "fresh" ? <ResourceFreshnessBadge state={monitor.state} /> : null}
                    </span>
                  </header>
                  <ResourceMiniSummary monitor={monitor} />
                  <small className="server-brief__sample">{serverItem.resource_monitor?.latest_success?.sampled_at ? tr("Local samples {0}", formatRelativeTime(serverItem.resource_monitor.latest_success.sampled_at)) : serverItem.resource_monitor?.latest_attempt?.error ?? tr("No successful background sample yet")}</small>
                </article>
              );
            })}
          </div>
        </section>
      </div>

      <section className="panel audit-strip">
        <header className="panel__header"><h2>{tr("Recent audit")}</h2><button className="text-link" onClick={() => onViewChange("settings")}>{tr("View all")}{" "}<ArrowRight size={14} /></button></header>
        {data.audit_events.length ? (
          <div className="audit-feed">
            {data.audit_events.slice(0, 6).map((event) => (
              <div className="audit-feed__row" key={event.id}>
                <i className={"audit-dot audit-dot--" + event.outcome} aria-hidden="true" />
                <time>{formatRelativeTime(event.timestamp)}</time>
                <code>{event.action}</code>
                <span className="audit-feed__detail" title={event.detail}>{event.detail}</span>
              </div>
            ))}
          </div>
        ) : <p className="muted-copy audit-feed__empty">{tr("No audit records yet")}</p>}
      </section>
    </div>
  );
}
