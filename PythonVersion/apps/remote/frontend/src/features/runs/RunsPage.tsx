import { useCallback, useEffect, useState } from "react";
import { Filter, Plus, Search } from "lucide-react";
import { toast } from "sonner";
import { api } from "../../api/client";
import { Button, EmptyState, LoadingBlock, Notice, PageHeader, RunStatusPill, Tabs } from "../../components/ui";
import { usePolling } from "../../hooks/usePolling";
import { formatDateTime } from "../../lib/format";
import type { RemoteRun } from "../../types";
import { NewRunDialog } from "./NewRunDialog";
import { RunDrawer } from "./RunDrawer";
import { tr } from "../../lib/i18n";

type RunFilter = "all" | "active" | "succeeded" | "failed";
const RUN_PAGE_SIZE = 100;

export function RunsPage({ pollSeconds, refreshNonce, openCreateSignal, initialTaskPath, selectedRunId, onSelectRun }: { pollSeconds: number; refreshNonce: number; openCreateSignal: number; initialTaskPath?: string | null; selectedRunId: string | null; onSelectRun: (id: string | null) => void }) {
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState<RunFilter>("all");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [createOpen, setCreateOpen] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => { setSearch(query.trim()); setPage(0); }, 300);
    return () => window.clearTimeout(timer);
  }, [query]);
  const loader = useCallback(() => api.runs.list({ limit: RUN_PAGE_SIZE + 1, offset: page * RUN_PAGE_SIZE, status: filter, search }), [refreshNonce, page, filter, search]);
  const { data: runs, loading, refreshing, error, refresh } = usePolling(loader, pollSeconds);
  const metadataLoader = useCallback(async () => {
    const [tasks, servers, settings] = await Promise.all([api.tasks.list(), api.servers.list(), api.settings.get()]);
    return { tasks, servers, settings };
  }, [refreshNonce, createOpen]);
  const { data: metadata, error: metadataError, loading: metadataLoading } = usePolling(metadataLoader, 60);
  const data = runs && metadata ? { ...metadata, runs } : null;
  const visibleRuns = runs?.slice(0, RUN_PAGE_SIZE) ?? [];
  const hasNext = (runs?.length ?? 0) > RUN_PAGE_SIZE;
  useEffect(() => { if (openCreateSignal > 0) setCreateOpen(true); }, [openCreateSignal]);

  const created = async (runId: string) => {
    setCreateOpen(false);
    toast.success(tr("Run queued"), { description: tr("The machine identity and environment are checked again before the code and task snapshots are frozen.") });
    await refresh();
    onSelectRun(runId);
  };

  if ((loading || metadataLoading) && !data) return <LoadingBlock label={tr("Reading the run history…")} />;
  if (!data) return <EmptyState title={tr("Cannot read the run records")} description={(error ?? metadataError)?.message ?? tr("Check that the local service is running.")} />;

  return (
    <div className="page">
      <PageHeader title={tr("Runs")} description={tr("Submit and monitor local and SSH experiments; view stage logs and execution artifacts")} actions={<Button tone="primary" onClick={() => setCreateOpen(true)} icon={<Plus size={15} />}>{tr("New run")}</Button>} />

      <section className="run-ledger panel">
        {error ? <div className="ledger-error"><Notice tone="danger"><strong>{tr("Run list refresh failed")}</strong><p>{error.message}</p></Notice></div> : null}
        <div className="ledger-toolbar">
          <label className="search-control"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("Search tasks, machines, run IDs or error codes")} /></label>
          <div className="filter-tabs"><Filter size={14} /><Tabs variant="segmented" ariaLabel={tr("Run status filter")} value={filter} onChange={(item) => { setFilter(item); setPage(0); }} items={[
            { value: "all", label: tr("All") },
            { value: "active", label: tr("In progress") },
            { value: "succeeded", label: tr("Completed") },
            { value: "failed", label: tr("Failed") },
          ]} /></div>
        </div>

        {visibleRuns.length ? (
          <>
            <div className="run-table-wrap">
              <table className="run-table">
                <thead><tr><th>{tr("Status")}</th><th>{tr("Task")}</th><th>{tr("Machines")}</th><th>{tr("Progress")}</th><th>{tr("Code version")}</th><th>{tr("Started / finished")}</th><th>{tr("Exit")}</th></tr></thead>
                <tbody>{visibleRuns.map((run) => <RunRow key={run.id} run={run} onClick={() => onSelectRun(run.id)} />)}</tbody>
              </table>
            </div>
            <div className="run-card-list">{visibleRuns.map((run) => <RunCard key={run.id} run={run} onClick={() => onSelectRun(run.id)} />)}</div>
            <footer className="ledger-pagination">
              <span>{tr("Page {0} · at most {1} per page", page + 1, RUN_PAGE_SIZE)}</span>
              <Button onClick={() => setPage((value) => Math.max(0, value - 1))} disabled={refreshing || page === 0}>{tr("Previous page")}</Button>
              <Button onClick={() => setPage((value) => value + 1)} disabled={refreshing || !hasNext}>{tr("Next page")}</Button>
            </footer>
          </>
        ) : <EmptyState title={tr("No matching records")} description={query || filter !== "all" ? tr("Adjust the search or filters.") : tr("After you create a local or SSH run, its complete evidence appears here.")} action={!query && filter === "all" ? <Button onClick={() => setCreateOpen(true)}>{tr("New run")}</Button> : undefined} />}
        {!visibleRuns.length && page > 0 ? <Button onClick={() => setPage(0)}>{tr("Back to the first page")}</Button> : null}
      </section>

      <NewRunDialog open={createOpen} initialTaskPath={initialTaskPath} tasks={data.tasks} servers={data.servers} settings={data.settings} onClose={() => setCreateOpen(false)} onCreated={(id) => void created(id)} />
      <RunDrawer key={selectedRunId ?? "closed"} runId={selectedRunId} pollSeconds={pollSeconds} refreshNonce={refreshNonce} onClose={() => onSelectRun(null)} onChanged={() => void refresh()} onDeleted={() => { onSelectRun(null); void refresh(); }} />
    </div>
  );
}

function RunRow({ run, onClick }: { run: RemoteRun; onClick: () => void }) {
  return (
    <tr tabIndex={0} onClick={onClick} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onClick(); }}>
      <td><RunStatusPill status={run.status} /></td>
      <td><strong>{run.task_label ?? run.task_path}</strong><small>{run.id}</small></td>
      <td><strong>{run.server_name}</strong><small>{run.max_parallel_runs} workers</small></td>
      <td><div className="table-progress"><span><i style={{ width: run.progress_percent + "%" }} /></span><small>{Math.round(run.progress_percent)}% · {run.completed_units}/{run.total_units}</small></div></td>
      <td><code>{run.git?.commit?.slice(0, 9) ?? "—"}</code><small>{run.git?.dirty ? "dirty snapshot" : "clean"}</small></td>
      <td><time>{formatDateTime(run.started_at ?? run.created_at)}</time><small>{run.finished_at ? tr("Finished ") + formatDateTime(run.finished_at) : run.current_message ?? tr("Waiting for an update")}</small></td>
      <td className={run.status === "failed" ? "is-error" : ""}><strong>{run.exit_code ?? "—"}</strong><small>{run.error_code ?? ""}</small></td>
    </tr>
  );
}

function RunCard({ run, onClick }: { run: RemoteRun; onClick: () => void }) {
  return (
    <button className="run-card" onClick={onClick}>
      <header><RunStatusPill status={run.status} /><time>{formatDateTime(run.started_at ?? run.created_at)}</time></header>
      <strong>{run.task_label ?? run.task_path}</strong><p>{run.server_name} · {run.max_parallel_runs} workers</p>
      <div className="table-progress"><span><i style={{ width: run.progress_percent + "%" }} /></span><small>{Math.round(run.progress_percent)}% · {run.current_message ?? tr("Waiting for an update")}</small></div>
      <footer><code>{run.git?.commit?.slice(0, 9) ?? "—"}{run.git?.dirty ? " · dirty" : ""}</code><span>{run.exit_code != null ? "exit " + run.exit_code : tr("Details ›")}</span></footer>
    </button>
  );
}
