import { useEffect, useMemo, useState } from "react";
import { FileJson2, Play, Plus, Save, Search, Trash2, Undo2 } from "lucide-react";
import { toast } from "sonner";
import { api } from "../../api/client";
import { Button, Badge, ConfirmDialog, EmptyState, LoadingBlock, Notice, PageHeader } from "../../components/ui";
import { formatDateTime, formatInteger } from "../../lib/format";
import type { TaskDocument, TaskFile } from "../../types";
import "./tasks.css";
import { tr } from "../../lib/i18n";

const starterJsonl = `{"max_parallel_runs":1,"default_problem":"GMPB","default_problem_params":{"Dimension":2,"PeakNumber":3,"ChangeFrequency":60,"EnvironmentNumber":2},"default_runs":1,"default_problem_seed":1,"default_algorithm_seed":101}
{"label":"smoke","algorithm":"ACFPSO","algorithm_params":{"PopulationSize":3}}
`;

export function TaskLibraryPage({ refreshNonce = 0, onRunTask }: { refreshNonce?: number; onRunTask: (taskPath: string) => void }) {
  const [tasks, setTasks] = useState<TaskFile[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [document, setDocument] = useState<TaskDocument | null>(null);
  const [filename, setFilename] = useState("");
  const [content, setContent] = useState("");
  const [original, setOriginal] = useState("");
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [pendingSwitch, setPendingSwitch] = useState<{ kind: "open"; path: string } | { kind: "new" } | null>(null);

  const dirty = content !== original || (creating && Boolean(content));
  const byteCount = useMemo(() => new TextEncoder().encode(content).byteLength, [content]);
  const objectCount = useMemo(() => content.split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith("#")).length, [content]);
  const visibleTasks = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return tasks;
    return tasks.filter((task) => task.path.toLowerCase().includes(keyword));
  }, [query, tasks]);

  const loadTasks = async (preferred?: string | null) => {
    setLoading(true);
    try {
      const items = await api.tasks.list();
      setTasks(items);
      const next = preferred && items.some((item) => item.path === preferred)
        ? preferred
        : selected && items.some((item) => item.path === selected)
          ? selected
          : items[0]?.path ?? null;
      if (next) await openTask(next, false);
      else if (!creating) {
        setSelected(null);
        setDocument(null);
        setFilename("");
        setContent("");
        setOriginal("");
      }
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : tr("Could not read the task library"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void loadTasks(); }, [refreshNonce]);

  const openTask = async (path: string, guard = true) => {
    if (guard && dirty) {
      setPendingSwitch({ kind: "open", path });
      return;
    }
    const task = tasks.find((item) => item.path === path);
    const taskFilename = task?.path.split("/").at(-1) ?? path.split("/").at(-1) ?? path;
    setLoading(true);
    try {
      const next = await api.tasks.get(taskFilename);
      setSelected(next.path);
      setDocument(next);
      setFilename(next.filename);
      setContent(next.content);
      setOriginal(next.content);
      setCreating(false);
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : tr("Could not read the task"));
    } finally {
      setLoading(false);
    }
  };

  const startNewTask = () => {
    setSelected(null);
    setDocument(null);
    setFilename("new_experiment.jsonl");
    setContent(starterJsonl);
    setOriginal("");
    setCreating(true);
    setError(null);
  };

  const newTask = () => {
    if (dirty) {
      setPendingSwitch({ kind: "new" });
      return;
    }
    startNewTask();
  };

  const confirmDiscard = () => {
    const action = pendingSwitch;
    setPendingSwitch(null);
    if (!action) return;
    if (action.kind === "new") startNewTask();
    else void openTask(action.path, false);
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const saved = creating
        ? await api.tasks.create(filename, content)
        : await api.tasks.update(filename, content);
      setDocument(saved);
      setSelected(saved.path);
      setFilename(saved.filename);
      setContent(saved.content);
      setOriginal(saved.content);
      setCreating(false);
      await loadTasks(saved.path);
      toast.success(tr("Task file validated and saved"), { description: tr("{0} test problems · {1} runs", saved.total_tasks, saved.total_runs) });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : tr("Task save failed"));
    } finally {
      setSaving(false);
    }
  };

  const remove = async () => {
    if (!document) return;
    setSaving(true);
    try {
      const deleted = await api.tasks.remove(document.filename);
      setDeleteOpen(false);
      setSelected(null);
      setDocument(null);
      setContent("");
      setOriginal("");
      await loadTasks(null);
      toast.success(tr("Task removed from the task library"), { description: tr("Can be restored from the Relay trash: {0}", deleted.recoverable_path) });
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : tr("Task deletion failed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page task-library-page">
      <PageHeader
        title={tr("Tasks")}
        description={tr("Edit tasks/*.jsonl of the repository; parameters and FE budgets are checked against the current registry before saving")}
        actions={<Button tone="primary" icon={<Plus size={15} />} onClick={newTask}>{tr("New task")}</Button>}
      />
      {error ? <Notice tone="danger"><strong>{tr("Task library operation failed")}</strong><p>{error}</p></Notice> : null}
      <div className="task-library-layout">
        <aside className="panel task-library-list">
          <header className="panel__header"><h2>{tr("Task file")}</h2><span className="panel-count">{visibleTasks.length}/{tasks.length}</span></header>
          <div className="task-library-list__search">
            <label className="search-control"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("Filter tasks/*.jsonl")} /></label>
          </div>
          <div className="task-library-list__scroll">
            {visibleTasks.map((task) => (
              <button key={task.path} className={selected === task.path ? "is-active" : ""} onClick={() => void openTask(task.path)}>
                <FileJson2 size={15} />
                <span><strong>{task.path.split("/").at(-1)}</strong><small>{task.size_bytes != null ? formatInteger(task.size_bytes) + " B" : ""}{task.modified_at ? ` · ${formatDateTime(task.modified_at)}` : ""}</small></span>
              </button>
            ))}
            {!visibleTasks.length && !creating ? <EmptyState title={tasks.length ? tr("No matching task files") : tr("The task library is empty")} description={tasks.length ? tr("Adjust the search keywords.") : tr("Create a JSONL task, or paste temporary JSONL directly when creating a run.")} /> : null}
          </div>
        </aside>

        <section className="panel task-editor">
          {loading && !content ? <LoadingBlock label={tr("Reading the task…")} /> : (
            <>
              <header className="panel__header task-editor__header">
                <div className="task-editor__title">
                  <h2 title={filename || tr("Choose a task")}>{filename || tr("Choose a task")}</h2>
                  <Badge tone={creating ? "info" : dirty ? "warning" : "success"}>{creating ? tr("New") : dirty ? tr("Unsaved") : tr("Validated")}</Badge>
                </div>
                <div className="task-editor__actions">
                  {!creating && document ? <Button icon={<Play size={14} />} onClick={() => onRunTask(document.path)}>{tr("Run this task")}</Button> : null}
                  {dirty ? <Button icon={<Undo2 size={14} />} onClick={() => setContent(original)}>{tr("Undo")}</Button> : null}
                  <Button tone="primary" icon={<Save size={14} />} disabled={!filename.trim() || !content.trim() || !dirty || saving} onClick={() => void save()}>{saving ? tr("Validating…") : tr("Validate and save")}</Button>
                  {!creating && document ? <Button tone="danger" icon={<Trash2 size={14} />} onClick={() => setDeleteOpen(true)}>{tr("Delete")}</Button> : null}
                </div>
              </header>
              <div className="task-editor__body">
                {document ? (
                  <div className={"task-editor__summary" + (dirty ? " is-stale" : "")}>
                    <span><small>{tr("Test problem")}</small><strong>{document.total_tasks}</strong></span>
                    <span><small>{tr("Total runs")}</small><strong>{formatInteger(document.total_runs)}</strong></span>
                    <span><small>{tr("Total FE")}</small><strong>{formatInteger(document.total_evaluations)}</strong></span>
                    <span><small>{dirty ? tr("SHA of the last validation") : "SHA-256"}</small><code title={document.sha256}>{document.sha256.slice(0, 12)}…</code></span>
                  </div>
                ) : null}
                {creating ? (
                  <label className="task-editor__filename"><span>{tr("File name")}</span><input value={filename} onChange={(event) => setFilename(event.target.value)} /><small>{tr("Must be a .jsonl file directly in tasks/.")}</small></label>
                ) : null}
                <label className="task-editor__source"><span>{tr("JSONL content")}</span><textarea spellCheck={false} value={content} onChange={(event) => setContent(event.target.value)} /><small>{tr("{0} non-empty entries · {1} UTF-8 bytes{2}", objectCount, formatInteger(byteCount), dirty ? tr(" · unsaved changes") : tr(" · saved"))}</small></label>
                {document?.warnings.map((warning) => <Notice key={warning} tone="warning"><strong>{tr("Budget notices")}</strong><p>{warning}</p></Notice>)}
              </div>
            </>
          )}
        </section>
      </div>
      <ConfirmDialog open={deleteOpen} title={tr("Remove task file")} message={tr("The task is moved to the local Relay trash; the frozen copies of submitted runs and past results are not affected.")} confirmLabel={tr("Move to trash")} destructive busy={saving} onConfirm={() => void remove()} onClose={() => setDeleteOpen(false)} />
      <ConfirmDialog
        open={pendingSwitch !== null}
        title={tr("Discard unsaved changes?")}
        message={pendingSwitch?.kind === "new" ? tr("The current task has unsaved changes; discard them and create a new task?") : tr("The current task has unsaved changes; discard them and open another task?")}
        confirmLabel={tr("Discard and continue")}
        destructive
        onConfirm={confirmDiscard}
        onClose={() => setPendingSwitch(null)}
      />
    </div>
  );
}
