import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  BarChart3,
  Check,
  ChevronDown,
  ChevronRight,
  Database,
  FileArchive,
  FileDown,
  FileJson2,
  FlaskConical,
  FolderCheck,
  History,
  Import,
  LineChart,
  RefreshCw,
  Search,
  Sigma,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { api } from "../../api/client";
import { Badge, Button, ConfirmDialog, Dialog, EmptyState, IconButton, LoadingBlock, Notice, PageHeader, Tabs } from "../../components/ui";
import { comparisonLabelPlan, type ComparisonLabelMode } from "../../lib/comparisonLabels";
import { formatBytes, formatDateTime, formatInteger, shortSha } from "../../lib/format";
import type {
  ResultCatalogItem,
  ResultCandidate,
  ResultRunset,
  ResultTaskEntry,
  RunResultsPreview,
  RunsetSamplesPreview,
  RunsetTrendPreview,
  SAAnalysis,
  SAImport,
  SAMethod,
  SAOutputFormat,
  SAResultPayload,
  ScalarMetricSummary,
} from "../../types";
import { tr, LOCALE } from "../../lib/i18n";

type AnalysisTab = "candidates" | "results" | "workspace" | "history";
type ResultDetailView = "runsets" | "plots";
type AnalysisDetailTab = "results" | "sources" | "technical";
type SelectedRunsetEntry = {
  resultId: string;
  resultName: string;
  bundleId: string;
  runset: ResultRunset;
};

const runsetSelectionKey = (resultId: string, runsetUid: string) => `${resultId}::${runsetUid}`;

const LazyPlotPage = lazy(async () => {
  const module = await import("../plots");
  return { default: module.PlotPage };
});

const methodLabels: Record<SAMethod, string> = {
  friedman: tr("Friedman + Nemenyi (paired, ≥3 groups)"),
  "signed-rank": tr("Wilcoxon signed-rank (paired, ≥2 groups)"),
  "rank-sum": tr("Wilcoxon rank-sum (unpaired, ≥2 groups)"),
};

const formatMetricValue = (value: number | null | undefined): string => {
  if (value == null || !Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute < 1e-4 || absolute >= 1e5)) return value.toExponential(4);
  return value.toLocaleString(LOCALE, { maximumSignificantDigits: 7 });
};

const normalizedParams = (value: Record<string, unknown> | null | undefined): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value : {};

const stringifyParamValue = (value: unknown): string => {
  if (value == null) return String(value);
  if (typeof value === "object") {
    try { return JSON.stringify(value); }
    catch { return tr("[complex value]"); }
  }
  return String(value);
};

const compactJson = (value: Record<string, unknown> | null | undefined): string => {
  const entries = Object.entries(normalizedParams(value));
  if (!entries.length) return tr("Default parameters");
  return entries.slice(0, 3).map(([key, item]) => `${key}=${stringifyParamValue(item)}`).join(" · ") + (entries.length > 3 ? " · …" : "");
};

const fullJson = (value: Record<string, unknown> | null | undefined): string => {
  try { return JSON.stringify(normalizedParams(value), null, 2); }
  catch { return tr("These parameters cannot be displayed"); }
};

const stableJson = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? String(value);
};

// Execution backends are provenance, not benchmark science.  Keep the full
// problem_params object in the UI, but exclude this key from SA compatibility
// checks so CPU and GPU/auto runs of the same problem can be compared.
const saExecutionProblemParams = new Set(["EvaluationDevice"]);
const saScientificProblemParams = (value: Record<string, unknown> | null | undefined): Record<string, unknown> =>
  Object.fromEntries(Object.entries(normalizedParams(value)).filter(([key]) => !saExecutionProblemParams.has(key)));
const saProblemParamsKey = (value: Record<string, unknown> | null | undefined): string =>
  stableJson(saScientificProblemParams(value));

const asResultPayload = (analysis: SAAnalysis | null): SAResultPayload | null => {
  if (!analysis || !analysis.result || !("analyses" in analysis.result)) return null;
  return analysis.result as SAResultPayload;
};

export function AnalysisPage({
  refreshNonce,
  requestedTab = "candidates",
  onRequestedTabChange,
  openPlotComposerSignal = 0,
}: {
  refreshNonce: number;
  requestedTab?: AnalysisTab;
  onRequestedTabChange?: (tab: AnalysisTab) => void;
  openPlotComposerSignal?: number;
}) {
  const [tab, setTab] = useState<AnalysisTab>(requestedTab);
  const [resultWorkspaceRevision, setResultWorkspaceRevision] = useState(0);
  const [candidates, setCandidates] = useState<ResultCandidate[]>([]);
  const [catalog, setCatalog] = useState<ResultCatalogItem[]>([]);
  const [imports, setImports] = useState<SAImport[]>([]);
  const [analyses, setAnalyses] = useState<SAAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null);
  const [bundleImportOpen, setBundleImportOpen] = useState(false);
  const [bundleArchive, setBundleArchive] = useState<File | null>(null);
  const [bundleImporting, setBundleImporting] = useState(false);
  const bundleArchiveInput = useRef<HTMLInputElement>(null);

  const refreshWorkspace = useCallback(async (quiet = false) => {
    if (!quiet) setRefreshing(true);
    try {
      const [nextCandidates, nextCatalog, nextImports, nextAnalyses] = await Promise.all([
        api.results.candidates(),
        api.results.catalog(),
        api.sa.imports(),
        api.sa.analyses(),
      ]);
      setCandidates(nextCandidates);
      setCatalog(nextCatalog);
      setImports(nextImports);
      setAnalyses(nextAnalyses);
      setSelectedResultId((current) => current && nextCatalog.some((item) => item.result_id === current) ? current : nextCatalog[0]?.result_id ?? null);
      setSelectedAnalysisId((current) => current && nextAnalyses.some((item) => item.id === current) ? current : nextAnalyses[0]?.id ?? null);
      setResultWorkspaceRevision((current) => current + 1);
      setError(null);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError : new Error(tr("Could not read the result workspace")));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { void refreshWorkspace(true); }, [refreshNonce, refreshWorkspace]);
  useEffect(() => { setTab(requestedTab); }, [requestedTab]);

  const importedCount = imports.filter((item) => item.source_available !== false).length;
  const completedAnalyses = analyses.filter((item) => item.status === "completed").length;

  const selectTab = (nextTab: AnalysisTab) => {
    setTab(nextTab);
    const nextHash = nextTab === "candidates" ? "#/analysis" : `#/${nextTab}`;
    if (window.location.hash !== nextHash) window.history.replaceState(null, "", nextHash);
    onRequestedTabChange?.(nextTab);
  };

  const uploadExternalArchive = async () => {
    if (!bundleArchive || bundleImporting) return;
    setBundleImporting(true);
    try {
      const result = await api.results.uploadArchive(bundleArchive);
      const summary = tr("Imported {0} runsets; skipped {1} duplicates; {2} failed validation.", result.imported_count, result.skipped_count, result.failed_count);
      if (result.failed_count) {
        const firstFailure = result.failed[0];
        toast.warning(tr("Result ZIP checked; some results were not added"), {
          description: `${summary}${firstFailure ? tr(" {0}: {1}", firstFailure.display_name, firstFailure.message) : ""}`,
        });
      } else {
        toast.success(tr("Result ZIP checked"), { description: summary });
      }
      setBundleImportOpen(false);
      setBundleArchive(null);
      if (bundleArchiveInput.current) bundleArchiveInput.current.value = "";
      const selectedId = result.imported[0]?.result.result_id ?? result.skipped.find((item) => item.duplicate_of_result_id)?.duplicate_of_result_id;
      if (selectedId) {
        setSelectedResultId(selectedId);
        selectTab("results");
      }
      await refreshWorkspace(true);
    } catch (nextError) {
      toast.error(tr("Result ZIP upload or validation failed"), { description: nextError instanceof Error ? nextError.message : tr("Unknown error") });
    } finally {
      setBundleImporting(false);
    }
  };

  const chooseBundleArchive = (file: File | null) => {
    if (file && !file.name.toLowerCase().endsWith(".zip")) {
      toast.error(tr("Choose a ZIP file"), { description: tr("External results must be standard EDOLAB result ZIP files.") });
      setBundleArchive(null);
      if (bundleArchiveInput.current) bundleArchiveInput.current.value = "";
      return;
    }
    setBundleArchive(file);
  };

  const closeBundleImport = () => {
    if (bundleImporting) return;
    setBundleImportOpen(false);
    setBundleArchive(null);
    if (bundleArchiveInput.current) bundleArchiveInput.current.value = "";
  };

  if (loading) return <LoadingBlock label={tr("Setting up the results and significance analysis workspace…")} />;

  return (
    <div className="page analysis-page">
      <PageHeader
        title={tr("Results")}
        description={tr("Import experiment results explicitly to preview metrics, plot them and test statistical significance")}
        actions={<><Button onClick={() => setBundleImportOpen(true)} icon={<Import size={15} />}>{tr("Import external results")}</Button><Button onClick={() => void refreshWorkspace()} disabled={refreshing} icon={<RefreshCw size={15} className={refreshing ? "is-spinning" : ""} />}>{refreshing ? tr("Refreshing…") : tr("Refresh workspace")}</Button></>}
      />

      {error ? <Notice tone="danger"><strong>{tr("Workspace refresh failed")}</strong><p>{error.message}</p></Notice> : null}

      <Tabs
        variant="underline"
        className="analysis-tabs"
        ariaLabel={tr("Results and analysis views")}
        value={tab}
        onChange={selectTab}
        items={[
          { value: "candidates", label: tr("Candidates"), count: candidates.length, icon: <FolderCheck size={14} />, title: tr("{0} run results can be imported", candidates.length) },
          { value: "results", label: tr("Imported results"), count: catalog.length, icon: <BarChart3 size={14} />, title: tr("{0} results imported by the user", catalog.length) },
          { value: "workspace", label: tr("SA workbench"), count: importedCount, icon: <FlaskConical size={14} />, title: tr("{0} runsets in the separate SA candidate list", importedCount) },
          { value: "history", label: tr("Analysis history"), count: `${completedAnalyses}/${analyses.length}`, icon: <History size={14} />, title: tr("{0} analyses completed, {1} running or failed", completedAnalyses, analyses.length - completedAnalyses) },
        ]}
      />

      {tab === "candidates" ? (
        <ResultCandidates
          candidates={candidates}
          onImportExternal={() => setBundleImportOpen(true)}
          onImported={(resultId) => {
            setSelectedResultId(resultId);
            void refreshWorkspace(true);
            selectTab("results");
          }}
        />
      ) : null}
      {tab === "results" ? (
        <ResultExplorer
          catalog={catalog}
          selectedResultId={selectedResultId}
          onSelectResult={setSelectedResultId}
          sourceRefreshToken={resultWorkspaceRevision}
          openPlotComposerSignal={openPlotComposerSignal}
          onWorkspaceChanged={() => void refreshWorkspace(true)}
          onImported={() => { void refreshWorkspace(true); selectTab("workspace"); }}
        />
      ) : null}
      {tab === "workspace" ? (
        <SAWorkspace
          imports={imports}
          onChanged={() => void refreshWorkspace(true)}
          onAnalysisCreated={(analysis) => {
            setSelectedAnalysisId(analysis.id);
            void refreshWorkspace(true);
            selectTab("history");
          }}
        />
      ) : null}
      {tab === "history" ? (
        <AnalysisHistory
          analyses={analyses}
          selectedAnalysisId={selectedAnalysisId}
          onSelectAnalysis={setSelectedAnalysisId}
          onChanged={() => void refreshWorkspace(true)}
        />
      ) : null}

      <Dialog
        open={bundleImportOpen}
        title={tr("Import external results")}
        description={tr("Upload a standard EDOLAB result ZIP from the computer you are browsing on; the ZIP may hold one or more results.")}
        onClose={closeBundleImport}
        footer={<><Button onClick={closeBundleImport} disabled={bundleImporting}>{tr("Cancel")}</Button><Button tone="primary" onClick={() => void uploadExternalArchive()} disabled={bundleImporting || !bundleArchive}>{bundleImporting ? tr("Uploading and checking…") : tr("Upload and import")}</Button></>}
      >
        <div className="result-upload-picker">
          <input ref={bundleArchiveInput} className="result-upload-picker__input" type="file" accept=".zip,application/zip,application/x-zip-compressed" onChange={(event) => chooseBundleArchive(event.target.files?.[0] ?? null)} />
          <FileArchive size={28} />
          <div><strong>{bundleArchive ? bundleArchive.name : tr("Choose an EDOLAB result ZIP")}</strong><p>{bundleArchive ? tr("{0} · waiting for upload and check", formatBytes(bundleArchive.size)) : tr("Choose a file on the computer you are browsing on; no server path is needed.")}</p></div>
          <Button onClick={() => bundleArchiveInput.current?.click()} disabled={bundleImporting}>{bundleArchive ? tr("Choose another") : tr("Choose ZIP")}</Button>
        </div>
        <Notice tone="info"><strong>{tr("Checked runset by runset")}</strong><p>{tr("The ZIP, manifest, summary.sqlite and trend.sqlite are validated first; then every runset is compared by its full parameters, run samples, statistics and trends. Duplicate runsets are skipped and only new ones are written; the numbers of imported, skipped and failed runsets are reported at the end.")}</p></Notice>
      </Dialog>
    </div>
  );
}

function ResultCandidates({
  candidates,
  onImportExternal,
  onImported,
}: {
  candidates: ResultCandidate[];
  onImportExternal: () => void;
  onImported: (resultId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [importingIds, setImportingIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    setSelectedIds((current) => new Set([...current].filter((id) => candidates.some((item) => item.candidate_id === id && item.available && !item.imported))));
  }, [candidates]);

  const visibleCandidates = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return candidates.filter((item) => !normalized || [
      item.display_name,
      item.task_name,
      item.task_path,
      item.run_id,
      item.server_name,
      item.bundle_relative_dir,
    ].join(" ").toLowerCase().includes(normalized));
  }, [candidates, query]);
  const visibleImportable = visibleCandidates.filter((item) => item.available && !item.imported);
  const allVisibleSelected = visibleImportable.length > 0 && visibleImportable.every((item) => selectedIds.has(item.candidate_id));
  const busy = importingIds.size > 0;

  const toggleCandidate = (candidateId: string) => setSelectedIds((current) => {
    const next = new Set(current);
    if (next.has(candidateId)) next.delete(candidateId); else next.add(candidateId);
    return next;
  });

  const toggleVisible = () => setSelectedIds((current) => {
    const next = new Set(current);
    for (const item of visibleImportable) {
      if (allVisibleSelected) next.delete(item.candidate_id);
      else next.add(item.candidate_id);
    }
    return next;
  });

  const importCandidates = async (candidateIds: string[]) => {
    const unique = [...new Set(candidateIds)].filter((id) => candidates.some((item) => item.candidate_id === id && item.available && !item.imported));
    if (!unique.length || busy) return;
    setImportingIds(new Set(unique));
    let lastResultId: string | null = null;
    let importedRunsetCount = 0;
    let skippedRunsetCount = 0;
    let failedRunsetCount = 0;
    const failures: string[] = [];
    for (const candidateId of unique) {
      const candidate = candidates.find((item) => item.candidate_id === candidateId);
      try {
        const outcome = await api.results.importCandidate(candidateId, candidate?.display_name);
        importedRunsetCount += outcome.imported_count;
        skippedRunsetCount += outcome.skipped_count;
        failedRunsetCount += outcome.failed_count;
        lastResultId = outcome.imported[0]?.result.result_id ?? outcome.skipped.find((item) => item.duplicate_of_result_id)?.duplicate_of_result_id ?? lastResultId;
        if (outcome.failed_count && outcome.failed[0]) failures.push(tr("{0}: {1}", candidate?.display_name ?? candidateId, outcome.failed[0].message));
      } catch (error) {
        failures.push(tr("{0}: {1}", candidate?.display_name ?? candidateId, error instanceof Error ? error.message : tr("Unknown error")));
      }
    }
    setImportingIds(new Set());
    setSelectedIds((current) => new Set([...current].filter((id) => !unique.includes(id))));
    if (importedRunsetCount || skippedRunsetCount || failedRunsetCount) {
      const description = tr("Imported {0} runsets; skipped {1} duplicates; {2} failed validation.", importedRunsetCount, skippedRunsetCount, failedRunsetCount);
      if (failedRunsetCount) toast.warning(tr("Candidates checked; some runsets were not added"), { description });
      else toast.success(tr("Candidates checked"), { description });
    }
    if (failures.length) toast.error(tr("Some candidates could not be imported"), { description: failures.join(tr("; ")) });
    if (lastResultId) onImported(lastResultId);
  };

  if (!candidates.length) return (
    <EmptyState
      title={tr("No candidates yet")}
      description={tr("Standard results of completed and validated runs appear here automatically, but they are not added to the imported results by themselves. External results can also be imported directly.")}
      action={<Button onClick={onImportExternal} icon={<Import size={15} />}>{tr("Import external results")}</Button>}
    />
  );

  return (
    <section className="result-candidates panel">
      <header className="result-candidates__header">
        <div><h2>{tr("Candidates")}</h2><p>{tr("Validated results of runs; once imported explicitly they enter the managed result library.")}</p></div>
        <Button onClick={onImportExternal} icon={<Import size={14} />}>{tr("Import external results")}</Button>
      </header>
      <div className="result-candidates__toolbar">
        <label className="search-control"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("Search by run name, machine, run ID or result folder")} /></label>
        <button className="text-link" disabled={!visibleImportable.length || busy} onClick={toggleVisible}>{allVisibleSelected ? tr("Deselect visible candidates") : tr("Select visible candidates")}</button>
        <span>{tr("{0} selected", selectedIds.size)}</span>
        <Button tone="primary" disabled={!selectedIds.size || busy} onClick={() => void importCandidates([...selectedIds])} icon={<Import size={14} />}>{busy ? tr("Copying and validating…") : tr("Import selected candidates")}</Button>
      </div>
      <div className="result-candidates__table-wrap">
        <table className="result-data-table result-candidates__table">
          <thead><tr><th>{tr("Select")}</th><th>{tr("Run result")}</th><th>{tr("Machines")}</th><th>{tr("Result folder")}</th><th>{tr("Status")}</th><th>{tr("Actions")}</th></tr></thead>
          <tbody>{visibleCandidates.map((item) => {
            const importable = item.available && !item.imported;
            const importing = importingIds.has(item.candidate_id);
            return <tr key={item.candidate_id}>
              <td><label className="runset-select" title={importable ? tr("Select to import explicitly") : undefined}><input aria-label={tr("Select candidate {0}", item.display_name)} type="checkbox" checked={selectedIds.has(item.candidate_id)} disabled={!importable || busy} onChange={() => toggleCandidate(item.candidate_id)} /><span /></label></td>
              <td><strong>{item.display_name}</strong><small>{item.run_id} · {item.task_path}</small></td>
              <td><strong>{item.server_name}</strong><small>{tr("{0} · finished {1}", item.target_kind, formatDateTime(item.completed_at ?? item.created_at))}</small></td>
              <td><code>{item.bundle_relative_dir}</code><small>{tr("{0} / {1} run units", formatInteger(item.completed_units), formatInteger(item.total_units))}</small></td>
              <td>{item.imported ? <Badge tone="success" icon={<Check size={12} />}>{tr("Imported")}</Badge> : item.available ? <Badge tone="info">{tr("Not imported")}</Badge> : <Badge tone="danger" icon={<X size={12} />}>{tr("Source unavailable")}</Badge>}</td>
              <td><Button tone="quiet" disabled={!importable || busy} onClick={() => void importCandidates([item.candidate_id])} icon={<Import size={13} />}>{item.imported ? tr("Imported") : importing ? tr("Importing…") : tr("Import")}</Button></td>
            </tr>;
          })}</tbody>
        </table>
        {!visibleCandidates.length ? <EmptyState title={tr("No matching candidates")} description={tr("Search by another run name, machine or result folder.")} /> : null}
      </div>
    </section>
  );
}

function ResultDrawer({ open, title, description, onClose, children, footer }: { open: boolean; title: string; description?: string; onClose: () => void; children: ReactNode; footer?: ReactNode }) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) return null;
  return (
    <div className="drawer-layer result-drawer-layer" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <section className="result-detail-drawer" role="dialog" aria-modal="true" aria-label={title}>
        <header className="result-detail-drawer__header">
          <div><h2>{title}</h2>{description ? <p>{description}</p> : null}</div>
          <IconButton aria-label={tr("Close details")} onClick={onClose}><X size={18} /></IconButton>
        </header>
        <div className="result-detail-drawer__body">{children}</div>
        {footer ? <footer className="result-detail-drawer__footer">{footer}</footer> : null}
      </section>
    </div>
  );
}

function ResultExplorer({
  catalog,
  selectedResultId,
  onSelectResult,
  sourceRefreshToken,
  openPlotComposerSignal,
  onWorkspaceChanged,
  onImported,
}: {
  catalog: ResultCatalogItem[];
  selectedResultId: string | null;
  onSelectResult: (id: string) => void;
  sourceRefreshToken: number;
  openPlotComposerSignal: number;
  onWorkspaceChanged: () => void;
  onImported: () => void;
}) {
  const [query, setQuery] = useState("");
  const [preview, setPreview] = useState<RunResultsPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<Error | null>(null);
  const [selectedRunsets, setSelectedRunsets] = useState<Map<string, SelectedRunsetEntry>>(new Map());
  const [selectedResults, setSelectedResults] = useState<Set<string>>(new Set());
  const [detailView, setDetailView] = useState<ResultDetailView>(openPlotComposerSignal ? "plots" : "runsets");
  const [plotSourceId, setPlotSourceId] = useState<string | undefined>();
  const [plotSelectionSignal, setPlotSelectionSignal] = useState(0);
  const [taskDrawerOpen, setTaskDrawerOpen] = useState(false);
  const [runsetDrawerUid, setRunsetDrawerUid] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [duplicateConfirmOpen, setDuplicateConfirmOpen] = useState(false);
  const [deleteTargets, setDeleteTargets] = useState<ResultCatalogItem[]>([]);
  const [deleting, setDeleting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportConfirmOpen, setExportConfirmOpen] = useState(false);
  const previewRequestId = useRef(0);

  const loadPreview = useCallback(async (resultId: string) => {
    const requestId = ++previewRequestId.current;
    setPreviewLoading(true);
    setPreviewError(null);
    setTaskDrawerOpen(false);
    setRunsetDrawerUid(null);
    try {
      const nextPreview = await api.results.preview(resultId);
      if (requestId !== previewRequestId.current) return;
      setPreview(nextPreview);
      setPlotSourceId(undefined);
    } catch (nextError) {
      if (requestId !== previewRequestId.current) return;
      setPreview(null);
      setPreviewError(nextError instanceof Error ? nextError : new Error(tr("Could not read the result preview")));
    } finally {
      if (requestId === previewRequestId.current) setPreviewLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedResultId) void loadPreview(selectedResultId);
    else {
      previewRequestId.current += 1;
      setPreview(null);
      setPreviewLoading(false);
      setTaskDrawerOpen(false);
      setRunsetDrawerUid(null);
    }
  }, [loadPreview, selectedResultId]);

  useEffect(() => {
    if (!openPlotComposerSignal) return;
    setDetailView("plots");
  }, [openPlotComposerSignal]);

  useEffect(() => {
    setSelectedResults((current) => new Set([...current].filter((id) => catalog.some((item) => item.result_id === id))));
    setSelectedRunsets((current) => {
      const availableResultIds = new Set(catalog.map((item) => item.result_id));
      const next = new Map([...current].filter(([, item]) => availableResultIds.has(item.resultId)));
      return next.size === current.size ? current : next;
    });
  }, [catalog]);

  useEffect(() => () => { previewRequestId.current += 1; }, []);

  const visibleCatalog = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return catalog.filter((item) => !normalized || [item.display_name, item.task_name, item.result_id, item.run_id, item.bundle_id, item.source_kind].join(" ").toLowerCase().includes(normalized));
  }, [catalog, query]);

  const selectedRunsetEntries = useMemo(() => [...selectedRunsets.values()], [selectedRunsets]);
  const selectedRunsetGroups = useMemo(() => {
    const groups = new Map<string, { resultId: string; resultName: string; bundleId: string; runsets: ResultRunset[] }>();
    for (const item of selectedRunsetEntries) {
      const group = groups.get(item.resultId) ?? {
        resultId: item.resultId,
        resultName: item.resultName,
        bundleId: item.bundleId,
        runsets: [],
      };
      group.runsets.push(item.runset);
      groups.set(item.resultId, group);
    }
    return [...groups.values()];
  }, [selectedRunsetEntries]);
  const selectedTaskCount = useMemo(() => new Set(selectedRunsetEntries.map((item) => `${item.resultId}::${item.runset.task_uid}`)).size, [selectedRunsetEntries]);
  const bundleSelectionCounts = useMemo(() => new Map(selectedRunsetGroups.map((item) => [item.resultId, item.runsets.length])), [selectedRunsetGroups]);
  const currentSelectedRunsets = useMemo(() => new Set(
    preview?.runsets
      .filter((item) => selectedRunsets.has(runsetSelectionKey(preview.result_id, item.runset_uid)))
      .map((item) => item.runset_uid) ?? [],
  ), [preview, selectedRunsets]);

  const toggleRunset = (uid: string) => {
    if (!preview) return;
    const runset = preview.runsets.find((item) => item.runset_uid === uid);
    if (!runset) return;
    const key = runsetSelectionKey(preview.result_id, uid);
    setSelectedRunsets((current) => {
      const next = new Map(current);
      if (next.has(key)) next.delete(key);
      else next.set(key, {
        resultId: preview.result_id,
        resultName: preview.display_name || preview.task_name,
        bundleId: preview.validation.bundle_id,
        runset,
      });
      return next;
    });
  };

  const completeRunsets = preview?.runsets.filter((item) => item.status === "completed" && item.completed_runs === item.expected_runs) ?? [];
  const allCompleteSelected = completeRunsets.length > 0 && completeRunsets.every((item) => currentSelectedRunsets.has(item.runset_uid));
  const activeRunset = preview?.runsets.find((item) => item.runset_uid === runsetDrawerUid) ?? null;
  const allResultsSelected = visibleCatalog.length > 0 && visibleCatalog.every((item) => selectedResults.has(item.result_id));

  const toggleResult = (resultId: string) => setSelectedResults((current) => {
    const next = new Set(current);
    if (next.has(resultId)) next.delete(resultId); else next.add(resultId);
    return next;
  });

  const toggleVisibleResults = () => setSelectedResults((current) => {
    const next = new Set(current);
    for (const item of visibleCatalog) {
      if (allResultsSelected) next.delete(item.result_id);
      else next.add(item.result_id);
    }
    return next;
  });

  const importSelected = async (allowDuplicates = false) => {
    if (!selectedRunsetEntries.length) return;
    setImporting(true);
    try {
      const result = await api.sa.importSelections(selectedRunsetGroups.map((group) => ({
        result_id: group.resultId,
        runset_uids: group.runsets.map((item) => item.runset_uid),
      })), allowDuplicates);
      const skipped = result.skipped ? tr(", {0} incomplete runsets skipped", result.skipped) : "";
      toast.success(tr("Added to the SA candidates"), { description: tr("Across {0} bundles: {1} added, {2} already present{3}. The candidate list is separate from the result library.", selectedRunsetGroups.length, result.imported, result.already_imported, skipped) });
      setDuplicateConfirmOpen(false);
      onImported();
    } catch (error) {
      if (!allowDuplicates && error && typeof error === "object" && "code" in error && error.code === "sa_duplicate_confirmation_required") {
        setDuplicateConfirmOpen(true);
      } else {
        toast.error(tr("Could not add to the SA candidates"), { description: error instanceof Error ? error.message : tr("Unknown error") });
      }
    }
    finally { setImporting(false); }
  };

  const openPlot = (runsetUid?: string) => {
    if (runsetUid && preview) {
      setPlotSourceId(`${preview.result_id}::${runsetUid}`);
      setPlotSelectionSignal((current) => current + 1);
    }
    setDetailView("plots");
  };

  const exportSelectedRunsets = async () => {
    if (!selectedRunsetEntries.length || exporting) return;
    setExporting(true);
    try {
      const archive = await api.results.exportRunsets(selectedRunsetGroups.map((group) => ({
        result_id: group.resultId,
        runset_uids: group.runsets.map((item) => item.runset_uid),
      })));
      const url = URL.createObjectURL(archive.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = archive.filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setExportConfirmOpen(false);
      toast.success(tr("Runset export started"), { description: tr("{0} runsets (from {1} bundles, {2} tasks) packed into a standard result ZIP.", selectedRunsetEntries.length, selectedRunsetGroups.length, selectedTaskCount) });
    } catch (error) { toast.error(tr("Runset export failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setExporting(false); }
  };

  const removeResults = async () => {
    if (!deleteTargets.length || deleting) return;
    setDeleting(true);
    try {
      for (const item of deleteTargets) await api.results.remove(item.result_id);
      const removedIds = new Set(deleteTargets.map((item) => item.result_id));
      setSelectedResults((current) => new Set([...current].filter((id) => !removedIds.has(id))));
      setSelectedRunsets((current) => new Map([...current].filter(([, item]) => !removedIds.has(item.resultId))));
      setDeleteTargets([]);
      toast.success(tr("Deleted from the result library"), { description: tr("Only the managed copy in the result library is deleted; the original folders, run records and earlier SA history are unaffected.") });
      onWorkspaceChanged();
    } catch (error) { toast.error(tr("Result deletion failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setDeleting(false); }
  };

  if (!catalog.length) return <EmptyState title={tr("No imported results yet")} description={tr("First import run results from “Candidates”, or import external results. Only the copies in this list can be previewed, plotted, exported and added to SA.")} />;

  return (
    <section className="result-explorer">
      <aside className="result-catalog panel">
        <header className="result-catalog__header"><h2>{tr("Imported results")}</h2><span>{visibleCatalog.length} / {catalog.length}</span></header>
        <div className="result-catalog__controls">
          <label className="search-control result-catalog__search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("Search by name, source or bundle")} /></label>
          <div className="result-catalog__bulk"><button className="text-link" onClick={toggleVisibleResults}>{allResultsSelected ? tr("Deselect filtered results") : tr("Select all filtered results")}</button><span>{tr("{0} selected", selectedResults.size)}</span><IconButton title={tr("Delete selected from the result library")} aria-label={tr("Delete selected from the result library")} disabled={!selectedResults.size} onClick={() => setDeleteTargets(catalog.filter((item) => selectedResults.has(item.result_id)))}><Trash2 size={14} /></IconButton></div>
        </div>
        <div className="result-catalog__list">
          {visibleCatalog.map((item) => (
            <article key={item.result_id} className={(selectedResultId === item.result_id ? "is-active " : "") + (selectedResults.has(item.result_id) ? "is-selected" : "")}>
              <label className="runset-select" title={tr("Select for deletion")}><input aria-label={tr("Select result {0} for deletion", item.display_name)} type="checkbox" checked={selectedResults.has(item.result_id)} onChange={() => toggleResult(item.result_id)} /><span /></label>
              <button onClick={() => onSelectResult(item.result_id)}>
                <span className="result-catalog__title"><strong>{item.display_name || item.task_name}</strong><i>{item.bundle_status}</i></span>
                <small>{item.source_kind} · {item.result_id}</small>
                <span className="result-catalog__facts"><b>{item.completed_runs}/{item.expected_runs} runs</b><b>{item.runset_count} runsets</b><b>{formatDateTime(item.imported_at)}</b></span>
                {bundleSelectionCounts.get(item.result_id) ? <em title={tr("Runset selection kept across bundles")}><Check size={11} />{tr("Selected")}{" "}{bundleSelectionCounts.get(item.result_id)}</em> : null}
                {item.imported_runset_count ? <em title={tr("In the separate SA candidate list")}><Check size={11} />{tr("SA candidate")}{" "}{item.imported_runset_count}</em> : null}
              </button>
              <IconButton title={tr("Delete this result from the result library")} aria-label={tr("Delete result {0}", item.display_name)} onClick={() => setDeleteTargets([item])}><Trash2 size={13} /></IconButton>
            </article>
          ))}
          {!visibleCatalog.length ? <EmptyState title={tr("No matching results")} description={tr("Search by another task name, path or bundle ID.")} /> : null}
        </div>
      </aside>

      <div className="result-preview panel">
        {previewLoading && !preview ? <LoadingBlock label={tr("Reading the imported result (read-only)…")} /> : previewError ? <EmptyState title={tr("Result preview failed")} description={previewError.message} action={selectedResultId ? <Button onClick={() => void loadPreview(selectedResultId)}>{tr("Retry")}</Button> : undefined} /> : preview ? (
          <>
            <header className="result-preview__header">
              <div><h2>{preview.display_name || preview.task_name}</h2><p title={`result ${preview.result_id}${preview.source_run_id ? tr(" · from run {0}", preview.source_run_id) : ""}`}>{preview.validation.completed_runs}/{preview.validation.expected_runs} runs · {preview.validation.runset_count} Runsets · {preview.tasks.length} tasks</p></div>
              <div><Button tone="quiet" onClick={() => setTaskDrawerOpen(true)} icon={<FileJson2 size={14} />}>{tr("Source tasks ({0})", preview.tasks.length)}</Button><Button tone="quiet" disabled={!selectedRunsetEntries.length || exporting} onClick={() => setExportConfirmOpen(true)} icon={<FileDown size={14} />}>{exporting ? tr("Exporting…") : tr("Export runsets ({0})", selectedRunsetEntries.length)}</Button><Button tone="primary" disabled={!selectedRunsetEntries.length || importing} onClick={() => void importSelected()} icon={<Import size={14} />}>{importing ? tr("Adding…") : tr("Add to SA candidates ({0})", selectedRunsetEntries.length)}</Button></div>
            </header>
            <Tabs
              variant="underline"
              className="result-detail-tabs"
              ariaLabel={tr("Result detail views")}
              value={detailView}
              onChange={(next) => { if (next === "plots") openPlot(); else setDetailView(next); }}
              items={[
                { value: "runsets", label: tr("Runset results"), icon: <BarChart3 size={14} /> },
                { value: "plots", label: tr("Plots"), icon: <LineChart size={14} /> },
              ]}
            />
            <div className="result-preview__body">
              {detailView === "runsets" ? <section className="preview-section runset-section">
                <header><div><BarChart3 size={15} /><span><strong>Runsets</strong><small>{tr("{0} in this bundle · {2} selected across {1} bundles", preview.runsets.length, selectedRunsetGroups.length, selectedRunsetEntries.length)}</small></span></div><button className="text-link" disabled={!completeRunsets.length} onClick={() => {
                  setSelectedRunsets((current) => {
                    const next = new Map(current);
                    for (const item of completeRunsets) {
                      const key = runsetSelectionKey(preview.result_id, item.runset_uid);
                      if (allCompleteSelected) next.delete(key);
                      else next.set(key, { resultId: preview.result_id, resultName: preview.display_name || preview.task_name, bundleId: preview.validation.bundle_id, runset: item });
                    }
                    return next;
                  });
                }}>{allCompleteSelected ? tr("Clear this bundle") : tr("Select all in this bundle ({0})", completeRunsets.length)}</button></header>
                <RunsetTable
                  runsets={preview.runsets}
                  selectedRunsets={currentSelectedRunsets}
                  onToggle={toggleRunset}
                  onOpen={(uid) => setRunsetDrawerUid(uid)}
                  onPlot={(uid) => openPlot(uid)}
                />
              </section> : <Suspense fallback={<LoadingBlock label={tr("Loading the plot workspace of this result…")} />}><LazyPlotPage key={preview.result_id} api={api.plots} sourceRefreshToken={sourceRefreshToken} active embedded scopeRunId={preview.result_id} scopeTaskName={preview.task_name} initialSourceId={plotSourceId} selectionSignal={plotSelectionSignal} /></Suspense>}
            </div>
          </>
        ) : <EmptyState title={tr("Choose an imported result")} description={tr("Only results copied or imported into the result library are listed on the left; run records are never mixed in.")} />}
      </div>

      <ConfirmDialog open={deleteTargets.length > 0} title={tr("Delete {0} items from the result library?", deleteTargets.length)} message={tr("This deletes the managed copies in the result library. The original import folders, the run records and the frozen SA history are not deleted; active SA candidates that still refer to these results may become unavailable.")} confirmLabel={tr("Delete result copies")} destructive busy={deleting} onClose={() => setDeleteTargets([])} onConfirm={() => void removeResults()} />
      <ConfirmDialog open={duplicateConfirmOpen} title={tr("Add duplicate algorithm configurations?")} message={tr("The selection across {0} bundles contains the same algorithm with the same parameters. As in GUIMode, this usually adds no information to a statistical comparison; if you do want to compare different sources or runsets, confirm to add them anyway.", selectedRunsetGroups.length)} confirmLabel={tr("Add anyway")} busy={importing} onClose={() => setDuplicateConfirmOpen(false)} onConfirm={() => void importSelected(true)} />

      <Dialog
        open={exportConfirmOpen}
        wide
        title={tr("Confirm export of {0} runsets", selectedRunsetEntries.length)}
        description={tr("Selected from {0} bundles and {1} tasks. The ZIP holds only the runsets listed below.", selectedRunsetGroups.length, selectedTaskCount)}
        onClose={() => { if (!exporting) setExportConfirmOpen(false); }}
        footer={<><Button disabled={exporting} onClick={() => setExportConfirmOpen(false)}>{tr("Back to selection")}</Button><Button tone="primary" disabled={!selectedRunsetEntries.length || exporting} onClick={() => void exportSelectedRunsets()} icon={<FileDown size={14} />}>{exporting ? tr("Creating ZIP…") : tr("Confirm export ({0})", selectedRunsetEntries.length)}</Button></>}
      >
        <div className="runset-export-groups">
          {selectedRunsetGroups.map((group) => (
            <section className="runset-export-group" key={group.resultId}>
              <header><div><strong>{group.resultName}</strong><small title={group.bundleId}>bundle {shortSha(group.bundleId)} · {group.runsets.length} Runsets</small></div><button className="text-link" disabled={exporting} onClick={() => setSelectedRunsets((current) => new Map([...current].filter(([, item]) => item.resultId !== group.resultId)))}>{tr("Remove this bundle")}</button></header>
              <div>
                {group.runsets.map((runset) => (
                  <article key={runset.runset_uid}>
                    <div><strong>{runset.label}</strong><small>{runset.problem_name} · {runset.algorithm_name} · {runset.completed_runs}/{runset.expected_runs} runs</small></div>
                    <code title={runset.runset_uid}>{runset.runset_uid}</code>
                    <IconButton disabled={exporting} title={tr("Do not export {0}", runset.label)} aria-label={tr("Remove {0} {1} from the export", group.resultName, runset.label)} onClick={() => setSelectedRunsets((current) => { const next = new Map(current); next.delete(runsetSelectionKey(group.resultId, runset.runset_uid)); return next; })}><X size={13} /></IconButton>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      </Dialog>

      <ResultDrawer open={Boolean(preview && taskDrawerOpen)} title={tr("JSONL test problems")} description={preview ? tr("{0} test problems · {1}", preview.tasks.length, preview.task_path) : undefined} onClose={() => setTaskDrawerOpen(false)}>
        {preview ? <TaskEntryList tasks={preview.tasks} /> : null}
      </ResultDrawer>

      <ResultDrawer
        open={Boolean(preview && activeRunset)}
        title={activeRunset ? `${activeRunset.label} · ${activeRunset.algorithm_name}` : tr("Runset details")}
        description={activeRunset ? `${activeRunset.problem_name} · ${activeRunset.completed_runs}/${activeRunset.expected_runs} runs · ${activeRunset.status}` : undefined}
        onClose={() => setRunsetDrawerUid(null)}
        footer={activeRunset ? <><span className="result-drawer-selection-note">{currentSelectedRunsets.has(activeRunset.runset_uid) ? tr("Selected; can be exported or added to SA candidates across bundles") : tr("This runset is not selected")}</span><Button tone={currentSelectedRunsets.has(activeRunset.runset_uid) ? "secondary" : "primary"} disabled={activeRunset.status !== "completed" || activeRunset.completed_runs !== activeRunset.expected_runs} onClick={() => toggleRunset(activeRunset.runset_uid)}>{currentSelectedRunsets.has(activeRunset.runset_uid) ? tr("Deselect") : tr("Select this runset")}</Button></> : undefined}
      >
        {preview && activeRunset ? <RunsetDetail key={activeRunset.runset_uid} runId={preview.result_id} runset={activeRunset} /> : null}
      </ResultDrawer>
    </section>
  );
}

function TaskEntryList({ tasks }: { tasks: ResultTaskEntry[] }) {
  return (
    <>
      <Notice tone="info"><strong>{tr("{0} test problems", tasks.length)}</strong><p>{tr("The JSONL entries frozen in the result bundle. They are stored with the imported copy and are not affected by deleting run records or by later changes to local JSONL files.")}</p></Notice>
      <div className="task-entry-list">
        {tasks.map((task) => (
          <details key={task.task_uid}>
            <summary><span><strong>{task.label}</strong><small>{tr("{0} · {1} runs · JSONL line {2}", task.problem_name, task.run_number, task.source_line ?? "—")}</small></span><ChevronDown size={14} /></summary>
            <div className="task-entry-detail">
              <dl><div><dt>{tr("Problem")}</dt><dd>{task.problem_name}</dd></div><div><dt>{tr("Number of runs")}</dt><dd>{task.run_number}</dd></div><div><dt>{tr("Source file")}</dt><dd title={task.source_task_file ?? undefined}>{task.source_task_file ?? "—"}</dd></div><div><dt>{tr("Configuration hash")}</dt><dd title={task.task_config_hash}>{shortSha(task.task_config_hash)}</dd></div></dl>
              <section><small>{tr("Problem parameters")}</small><pre>{fullJson(task.problem_params)}</pre></section>
            </div>
          </details>
        ))}
      </div>
    </>
  );
}

function RunsetTable({ runsets, selectedRunsets, onToggle, onOpen, onPlot }: { runsets: ResultRunset[]; selectedRunsets: Set<string>; onToggle: (uid: string) => void; onOpen: (uid: string) => void; onPlot: (uid: string) => void }) {
  return (
    <div className="runset-table-scroll">
      <table className="runset-table">
        <thead><tr><th aria-label={tr("Runset selection")} /><th>{tr("Problem label")}</th><th>{tr("Problem and parameters")}</th><th>{tr("Algorithm and parameters")}</th><th>{tr("Run / SA status")}</th><th>{tr("Actions")}</th></tr></thead>
        <tbody>
          {runsets.map((runset) => {
            const complete = runset.status === "completed" && runset.completed_runs === runset.expected_runs;
            const selected = selectedRunsets.has(runset.runset_uid);
            const metricNames = (runset.scalar_metrics ?? []).map((metric) => metric.name).join(tr(", "));
            return (
              <tr
                key={runset.runset_uid}
                className={(selected ? "is-selected " : "") + (!complete ? "is-incomplete" : "")}
                tabIndex={0}
                onClick={() => onOpen(runset.runset_uid)}
                onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); onOpen(runset.runset_uid); } }}
              >
                <td className="runset-table__select" onClick={(event) => event.stopPropagation()}>
                  <label className="runset-select" title={complete ? (selected ? tr("Deselect") : tr("Select to export or add to SA candidates")) : tr("Incomplete run; cannot be selected")}><input aria-label={tr("Select runset {0} {1}", runset.label, runset.algorithm_name)} type="checkbox" checked={selected} disabled={!complete} onChange={() => onToggle(runset.runset_uid)} /><span /></label>
                </td>
                <td className="runset-table__label"><strong title={runset.label}>{runset.label}</strong></td>
                <td className="runset-table__params"><strong>{runset.problem_name}</strong><span title={fullJson(runset.problem_params)}>{compactJson(runset.problem_params)}</span></td>
                <td className="runset-table__params"><strong>{runset.algorithm_name}</strong><span title={fullJson(runset.algorithm_params)}>{compactJson(runset.algorithm_params)}</span></td>
                <td className="runset-table__result"><div><strong className={complete ? "is-complete" : ""}>{runset.completed_runs}/{runset.expected_runs} runs</strong>{runset.sa_import_id ? <span className="analysis-candidate-state is-imported" title={tr("This runset is in the SA candidate pool")}><Check size={11} />SA</span> : null}</div><small title={metricNames || undefined}>{tr("{0} summary · {1} trend", runset.scalar_metrics.length, runset.trend_metrics.length)}</small></td>
                <td className="runset-table__open"><IconButton title={tr("Add to the plot and switch to the plot view")} aria-label={tr("Plot {0}", runset.label)} disabled={!complete} onClick={(event) => { event.stopPropagation(); onPlot(runset.runset_uid); }}><LineChart size={14} /></IconButton><IconButton title={tr("Open the detail drawer")} aria-label={tr("View details of {0} {1}", runset.label, runset.algorithm_name)} onClick={(event) => { event.stopPropagation(); onOpen(runset.runset_uid); }}><ChevronRight size={15} /></IconButton></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RunsetDetail({ runId, runset }: { runId: string; runset: ResultRunset }) {
  const [samples, setSamples] = useState<RunsetSamplesPreview | null>(null);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [trendMetric, setTrendMetric] = useState(runset.trend_metrics[0]?.name ?? "");
  const [trend, setTrend] = useState<RunsetTrendPreview | null>(null);
  const [trendLoading, setTrendLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSamples = async () => {
    setSamplesLoading(true);
    setError(null);
    try {
      const offset = samples?.runs.length ?? 0;
      const next = await api.results.samples(runId, runset.runset_uid, { offset, limit: 200 });
      setSamples((current) => current ? { ...next, offset: 0, runs: [...current.runs, ...next.runs] } : next);
    } catch (nextError) { setError(nextError instanceof Error ? nextError.message : tr("Could not read the raw samples")); }
    finally { setSamplesLoading(false); }
  };

  const loadTrend = async (metricName = trendMetric) => {
    if (!metricName) return;
    setTrendLoading(true);
    setError(null);
    setTrendMetric(metricName);
    try { setTrend(await api.results.trend(runId, runset.runset_uid, metricName)); }
    catch (nextError) { setError(nextError instanceof Error ? nextError.message : tr("Could not read the trend")); }
    finally { setTrendLoading(false); }
  };

  const sampleMetrics = useMemo(() => [...new Set(samples?.runs.flatMap((item) => Object.keys(item.metrics)) ?? [])], [samples]);

  return (
    <div className="runset-detail-content">
      <section className="runset-detail-facts"><span><small>{tr("Problem")}</small><strong>{runset.problem_name}</strong></span><span><small>{tr("Algorithm")}</small><strong>{runset.algorithm_name}</strong></span><span><small>{tr("Complete runs")}</small><strong>{runset.completed_runs}/{runset.expected_runs}</strong></span><span><small>{tr("Summary metrics")}</small><strong>{runset.scalar_metrics.length}</strong></span><span><small>{tr("Trend metrics")}</small><strong>{runset.trend_metrics.length}</strong></span></section>
      <section className="runset-detail-section"><header><h3>{tr("Problem and algorithm parameters")}</h3><span>{tr("Actual parameters of this runset")}</span></header><div className="runset-parameter-grid"><section><small>{tr("Problem parameters")}</small><pre>{fullJson(runset.problem_params)}</pre></section><section><small>{tr("Algorithm parameters")}</small><pre>{fullJson(runset.algorithm_params)}</pre></section></div></section>
      <section className="runset-detail-section"><header><h3>{tr("Summary metrics")}</h3><span>{tr("Mean ± SE, median and direction from summary.sqlite")}</span></header>{runset.scalar_metrics.length ? <MetricSummaryTable metrics={runset.scalar_metrics} /> : <Notice tone="warning"><strong>{tr("No summary scalars")}</strong><p>{tr("This runset cannot be used for scalar significance analysis.")}</p></Notice>}</section>
      <section className="runset-detail-section"><header><h3>{tr("On-demand data")}</h3><span>{tr("Raw samples and trends are read only when requested")}</span></header><div className="runset-detail-section__body">
        <div className="lazy-result-toolbar">
          <Button tone="quiet" disabled={samplesLoading} onClick={() => void loadSamples()} icon={<Database size={13} />}>{samples ? samples.runs.length < samples.total ? tr("Load more samples ({0}/{1})", samples.runs.length, samples.total) : tr("{0} samples loaded", samples.total) : samplesLoading ? tr("Reading samples…") : tr("Read raw samples")}</Button>
          {runset.trend_metrics.length ? <div className="trend-loader"><select value={trendMetric} onChange={(event) => { setTrendMetric(event.target.value); setTrend(null); }}>{runset.trend_metrics.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.axis} · {item.sample_count}</option>)}</select><Button tone="quiet" disabled={trendLoading} onClick={() => void loadTrend()} icon={<LineChart size={13} />}>{trendLoading ? tr("Reading trend…") : tr("Preview trend")}</Button></div> : <small>{tr("No trend metrics")}</small>}
        </div>
        {error ? <Notice tone="danger"><strong>{tr("Could not read the on-demand data")}</strong><p>{error}</p></Notice> : null}
        {samples ? <RawSampleTable samples={samples} metrics={sampleMetrics} /> : null}
        {trend ? <TrendPreview trend={trend} /> : null}
      </div></section>
    </div>
  );
}

function MetricSummaryTable({ metrics }: { metrics: ScalarMetricSummary[] }) {
  return <div className="result-table-wrap"><table className="result-data-table metric-summary-table"><thead><tr><th>{tr("Metric")}</th><th>Mean ± SE</th><th>Median</th><th>n</th><th>{tr("Direction")}</th></tr></thead><tbody>{metrics.map((metric) => <tr key={metric.name}><td><strong>{metric.name}</strong></td><td>{formatMetricValue(metric.mean)} <span>± {formatMetricValue(metric.stderr)}</span></td><td>{formatMetricValue(metric.median)}</td><td>{metric.count}</td><td><i className={metric.direction === "lower" ? "is-lower" : "is-higher"}>{metric.direction === "lower" ? tr("Lower is better") : metric.direction === "higher" ? tr("Higher is better") : metric.direction}</i></td></tr>)}</tbody></table></div>;
}

function RawSampleTable({ samples, metrics }: { samples: RunsetSamplesPreview; metrics: string[] }) {
  return (
    <details className="lazy-result-block" open>
      <summary><span>{tr("Raw run samples")}</span><small>{tr("Showing {0} / {1}", samples.runs.length, samples.total)}</small><ChevronDown size={13} /></summary>
      <div className="result-table-wrap"><table className="result-data-table raw-sample-table"><thead><tr><th>#</th><th>Problem seed</th><th>Algorithm seed</th><th>FE</th><th>{tr("Duration")}</th>{metrics.map((metric) => <th key={metric}>{metric}</th>)}</tr></thead><tbody>{samples.runs.map((run) => <tr key={run.run_uid}><td>{run.run_index}</td><td>{run.problem_seed}</td><td>{run.algorithm_seed}</td><td>{formatInteger(run.fe)}</td><td>{run.runtime_seconds == null ? "—" : `${run.runtime_seconds.toFixed(2)}s`}</td>{metrics.map((metric) => <td key={metric}>{formatMetricValue(run.metrics[metric])}</td>)}</tr>)}</tbody></table></div>
    </details>
  );
}

function TrendPreview({ trend }: { trend: RunsetTrendPreview }) {
  const points = trend.mean.map((value, index) => ({ x: trend.key[index], y: value })).filter((item): item is { x: number; y: number } => Number.isFinite(item.x) && item.y != null && Number.isFinite(item.y));
  const width = 720;
  const height = 180;
  const padding = 24;
  const xValues = points.map((item) => item.x);
  const yValues = points.map((item) => item.y);
  const minX = Math.min(...xValues, 0);
  const maxX = Math.max(...xValues, 1);
  const minY = Math.min(...yValues, 0);
  const maxY = Math.max(...yValues, 1);
  const xRange = maxX - minX || 1;
  const yRange = maxY - minY || 1;
  const path = points.map((point, index) => {
    const x = padding + (point.x - minX) / xRange * (width - padding * 2);
    const y = height - padding - (point.y - minY) / yRange * (height - padding * 2);
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  return (
    <details className="lazy-result-block" open>
      <summary><span>{tr("{0} trend", trend.metric_name)}</span><small>{tr("{0} · showing {1}/{2} points", trend.axis, trend.mean.length, trend.total)}</small><ChevronDown size={13} /></summary>
      {points.length ? <div className="trend-preview-chart"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${trend.metric_name} mean trend`}><line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} /><line x1={padding} y1={padding} x2={padding} y2={height - padding} /><path d={path} /><text x={padding} y={height - 7}>{formatMetricValue(minX)}</text><text x={width - padding} y={height - 7} textAnchor="end">{formatMetricValue(maxX)}</text><text x={padding + 4} y={padding + 10}>{formatMetricValue(maxY)}</text><text x={padding + 4} y={height - padding - 5}>{formatMetricValue(minY)}</text></svg><div><span>Mean</span><small>{tr("Shaded band not drawn; the SE of every point is kept in the table below.")}</small></div></div> : <p className="muted-copy">{tr("No finite trend data on this page.")}</p>}
      <div className="result-table-wrap trend-data-table"><table className="result-data-table"><thead><tr><th>{trend.axis}</th><th>Mean</th><th>SE</th><th>n</th></tr></thead><tbody>{trend.key.slice(0, 60).map((key, index) => <tr key={`${key}-${index}`}><td>{formatMetricValue(key)}</td><td>{formatMetricValue(trend.mean[index])}</td><td>{formatMetricValue(trend.stderr[index])}</td><td>{trend.count[index]}</td></tr>)}</tbody></table>{trend.key.length > 60 ? <small>{tr("The table shows the first 60 points; the line uses all {0} loaded points.", trend.key.length)}</small> : null}</div>
    </details>
  );
}

function SAWorkspace({ imports, onChanged, onAnalysisCreated }: { imports: SAImport[]; onChanged: () => void; onAnalysisCreated: (analysis: SAAnalysis) => void }) {
  const availableImports = useMemo(() => imports.filter((item) => item.source_available !== false), [imports]);
  const templateImport = imports[0];
  const [method, setMethod] = useState<SAMethod>("friedman");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectedMetrics, setSelectedMetrics] = useState<Set<string>>(new Set());
  const [name, setName] = useState("");
  const [alpha, setAlpha] = useState("0.05");
  const [labelMode, setLabelMode] = useState<ComparisonLabelMode>("algorithm");
  const [formats, setFormats] = useState<Set<SAOutputFormat>>(new Set(["sqlite", "csv"]));
  const [creating, setCreating] = useState(false);
  const [clearOpen, setClearOpen] = useState(false);
  const [quickFiltering, setQuickFiltering] = useState(false);
  const [quickFilterDuplicateConfirmOpen, setQuickFilterDuplicateConfirmOpen] = useState(false);

  useEffect(() => {
    setSelectedIds((current) => new Set([...current].filter((id) => availableImports.some((item) => item.id === id))));
  }, [availableImports]);

  const minimum = method === "friedman" ? 3 : 2;
  const alphaValue = Number(alpha);
  const selectedImports = useMemo(() => imports.filter((item) => selectedIds.has(item.id) && item.source_available !== false), [imports, selectedIds]);
  const selectedMatch = useMemo(() => {
    if (!selectedImports.length) return { compatible: false, message: tr("Choose the SA candidates for this analysis by hand."), metrics: [] as string[], warnings: [] as string[] };
    const first = selectedImports[0];
    const paired = method !== "rank-sum";
    const compatible = selectedImports.every((item) => item.problem_name === first.problem_name
      && saProblemParamsKey(item.problem_params) === saProblemParamsKey(first.problem_params)
      && (!paired || item.completed_runs === first.completed_runs && item.pairing_signature === first.pairing_signature));
    const metricMaps = selectedImports.map((item) => new Map(item.scalar_metrics.map((metric) => [metric.name, metric.direction])));
    const metrics = [...(metricMaps[0]?.keys() ?? [])].filter((metric) => metricMaps.every((map) => map.get(metric) === metricMaps[0].get(metric)));
    const executionDevices = new Set(selectedImports
      .map((item) => item.problem_params?.EvaluationDevice)
      .filter((value) => value != null)
      .map((value) => stringifyParamValue(value)));
    return {
      compatible,
      metrics,
      warnings: [
        ...(selectedImports.some((item) => item.source_available === false) ? [tr("Some result sources are no longer available.")] : []),
        ...(executionDevices.size > 1 ? [tr("The execution device (EvaluationDevice) differs; it is not part of the problem match and is kept in the source audit.")] : []),
      ],
      message: compatible
        ? tr("The problem configuration{1} of the {0} selected items match.", selectedImports.length, paired ? tr(", run count and pairing signature") : "")
        : tr("The selected items do not meet the problem-configuration{1} requirement of {0}.", methodLabels[method], paired ? tr(" and pairing") : ""),
    };
  }, [method, selectedImports]);
  const labelPlan = useMemo(() => comparisonLabelPlan(selectedImports, labelMode), [labelMode, selectedImports]);
  const canCreate = selectedIds.size >= minimum && selectedMatch.compatible && selectedMetrics.size > 0 && name.trim() && alphaValue > 0 && alphaValue < 1 && formats.size > 0;

  useEffect(() => {
    setSelectedMetrics((current) => new Set([...current].filter((metric) => selectedMatch.metrics.includes(metric))));
  }, [selectedMatch.metrics]);

  useEffect(() => {
    if (labelPlan.mode !== labelMode) setLabelMode(labelPlan.mode);
  }, [labelMode, labelPlan.mode]);

  const toggleSelected = (id: string) => setSelectedIds((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const toggleMetric = (metric: string) => setSelectedMetrics((current) => {
    const next = new Set(current);
    if (next.has(metric)) next.delete(metric); else next.add(metric);
    return next;
  });
  const toggleFormat = (format: SAOutputFormat) => setFormats((current) => {
    const next = new Set(current);
    if (next.has(format)) next.delete(format); else next.add(format);
    return next;
  });

  const removeImport = async (id: string) => {
    try { await api.sa.removeImport(id); onChanged(); }
    catch (error) { toast.error(tr("Could not remove the candidate"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
  };

  const clearImports = async () => {
    try { await api.sa.clearImports(); setClearOpen(false); onChanged(); toast.success(tr("SA candidate list cleared"), { description: tr("The result library, run records and past analyses were not deleted.") }); }
    catch (error) { toast.error(tr("Clearing failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
  };

  const quickFilter = async (allowDuplicates = false) => {
    if (!imports.length || quickFiltering) return;
    setQuickFiltering(true);
    try {
      const result = await api.sa.quickFilter(allowDuplicates);
      const added = result.added;
      toast.success(added ? tr("{0} compatible candidates added", added) : tr("No new compatible candidates"), { description: tr("Taking the first item of the SA list as the template, only runsets of the result library with the same problem, run count and problem parameters are chosen; differences of execution device are kept in the source audit.") });
      setQuickFilterDuplicateConfirmOpen(false);
      onChanged();
    } catch (error) {
      if (!allowDuplicates && error && typeof error === "object" && "code" in error && error.code === "sa_duplicate_confirmation_required") {
        setQuickFilterDuplicateConfirmOpen(true);
      } else {
        toast.error(tr("Automatic selection failed"), { description: error instanceof Error ? error.message : tr("Unknown error") });
      }
    }
    finally { setQuickFiltering(false); }
  };

  const createAnalysis = async () => {
    if (!canCreate) return;
    setCreating(true);
    try {
      const analysis = await api.sa.createAnalysis({ name: name.trim(), method, import_ids: [...selectedIds], metrics: [...selectedMetrics], alpha: alphaValue, formats: [...formats], label_parameter: labelPlan.parameter });
      toast.success(tr("Significance analysis completed"), { description: tr("{0} metrics, {1} runsets; output and source snapshot archived.", analysis.metric_names.length, analysis.source_imports.length) });
      onAnalysisCreated(analysis);
    } catch (error) { toast.error(tr("Significance analysis failed"), { description: error instanceof Error ? error.message : tr("Unknown error") }); }
    finally { setCreating(false); }
  };

  if (!imports.length) return <EmptyState title={tr("The SA candidate list is empty")} description={tr("First choose at least one complete runset in “Result library” and click “Add to SA candidates”. The first candidate becomes the template for GUIMode-style automatic selection; results are never added to SA automatically.")} />;

  return (
    <section className="sa-workspace">
      <div className="sa-candidates panel">
        <header className="panel__header"><h2>{tr("SA candidate runsets")}</h2><Button tone="quiet" disabled={!imports.length} onClick={() => setClearOpen(true)} icon={<X size={13} />}>{tr("Clear list")}</Button></header>
        <div className="sa-candidate-toolbar">
          <label className="field"><span>{tr("Statistical method")}</span><select value={method} onChange={(event) => setMethod(event.target.value as SAMethod)}>{(Object.keys(methodLabels) as SAMethod[]).map((item) => <option key={item} value={item}>{methodLabels[item]}</option>)}</select></label>
          <div className="sa-template"><span>{tr("Compatibility template (first item)")}</span><strong>{templateImport ? `${templateImport.task_name} · ${templateImport.label} · ${templateImport.algorithm_name}` : "—"}</strong><small>{templateImport?.problem_name} · {templateImport?.completed_runs} runs · {compactJson(templateImport?.problem_params)}{templateImport?.source_available === false ? tr(" · source result deleted, using the frozen snapshot") : ""}</small></div>
          <Button onClick={() => void quickFilter()} disabled={quickFiltering || !imports.length} icon={<RefreshCw size={14} className={quickFiltering ? "is-spinning" : ""} />}>{quickFiltering ? tr("Selecting…") : tr("Select compatible items from the result library")}</Button>
        </div>
        <div className={"sa-match-banner " + (selectedMatch.compatible ? "is-compatible" : "is-incompatible")}><span>{selectedMatch.compatible ? <Check size={14} /> : <X size={14} />}</span><div><strong>{selectedMatch.compatible ? tr("{0} compatible runsets selected", selectedImports.length) : tr("The current selection cannot be analysed yet")}</strong><p>{tr("{0} · Changing the statistical method only re-checks the current selection; the candidate list is not changed.", selectedMatch.message)}</p></div></div>
        {selectedMatch.warnings.map((warning) => <Notice key={warning} tone="warning"><strong>{tr("Source version notice")}</strong><p>{warning}</p></Notice>)}
        <div className="sa-candidate-list">
          {imports.map((item) => {
            const available = item.source_available !== false;
            const template = item.id === templateImport?.id;
            return <article key={item.id} className={(selectedIds.has(item.id) ? "is-selected " : "") + (!available ? "is-unavailable" : "")}><label><input type="checkbox" checked={selectedIds.has(item.id)} disabled={!available} onChange={() => toggleSelected(item.id)} /><span /></label><div><span><strong>{item.task_name}</strong>{template ? <i>{tr("Compatibility template")}</i> : null}</span><p>{item.label} · {item.algorithm_name}</p><small>{item.problem_name} · {item.completed_runs} runs · {compactJson(item.problem_params)}</small></div><button onClick={() => void removeImport(item.id)} title={tr("Remove from the SA candidate list")}><Trash2 size={14} /></button></article>;
          })}
        </div>
      </div>

      <aside className="sa-config panel">
        <header className="panel__header"><h2>{tr("Analysis settings")}</h2><span className="panel-count">{tr("{0} selected", selectedIds.size)}</span></header>
        <div className="sa-config__body">
          <label className="field"><span>{tr("Analysis name")}</span><input maxLength={200} value={name} onChange={(event) => setName(event.target.value)} placeholder={tr("For example: Figure 4 · GMPB main experiment E_o")} /><small>{tr("The history and the output folder keep this name.")}</small></label>
          <label className="field"><span>{tr("Significance level α")}</span><input type="number" min="0.0001" max="0.9999" step="0.01" value={alpha} onChange={(event) => setAlpha(event.target.value)} /><small>{tr("Must lie between 0 and 1; 0.05 is usual.")}</small></label>
          <label className="field"><span>{tr("Statistics label")}</span><select value={labelPlan.mode} disabled={labelPlan.mixedAlgorithms || !labelPlan.parameterModes.length} onChange={(event) => setLabelMode(event.target.value as ComparisonLabelMode)}>{labelPlan.mixedAlgorithms || !labelPlan.parameterModes.length ? <option value="algorithm">{tr("Algorithm name (automatic)")}</option> : null}{labelPlan.parameterModes.map((option) => <option key={option} value={option}>{tr("Algorithm parameter · {0}", option.slice("algorithmParam:".length))}</option>)}</select><small>{labelPlan.mixedAlgorithms ? tr("The selected runsets hold several algorithms, so the tables are always grouped by algorithm name.") : labelPlan.parameterModes.length ? tr("Parameter sensitivity of a single algorithm: the columns show the values of the parameter {0}.", labelPlan.title) : tr("No common algorithm parameter varies in this selection; repeated algorithm names get a short runset ID.")}</small></label>

          <fieldset className="sa-option-group"><legend>{tr("Common scalar metrics of the selection")}</legend>{selectedMatch.metrics.length ? selectedMatch.metrics.map((metric) => <label key={metric}><input type="checkbox" checked={selectedMetrics.has(metric)} onChange={() => toggleMetric(metric)} /><span>{metric}</span></label>) : <p>{tr("The selection has no common scalar metric with a consistent direction.")}</p>}</fieldset>
          <fieldset className="sa-option-group"><legend>{tr("Output formats")}</legend>{(["sqlite", "csv", "xlsx", "latex"] as SAOutputFormat[]).map((format) => <label key={format}><input type="checkbox" checked={formats.has(format)} onChange={() => toggleFormat(format)} /><span>{format.toUpperCase()}</span></label>)}</fieldset>

          <div className="sa-method-note"><strong>{methodLabels[method]}</strong><p>{method === "friedman" ? tr("Requires the same problem parameters, the same number of runs and the same ordered problem seeds; the execution device is not part of the problem match. The Friedman test is followed by the Nemenyi post-hoc test and compact letter groups.") : method === "signed-rank" ? tr("Paired two-sample test: requires the same problem parameters, the same number of runs and the same ordered problem seeds; the execution device is not part of the problem match; with more than two groups all pairwise comparisons are Holm-corrected.") : tr("Unpaired two-sample test: requires only the same problem and problem parameters; the execution device is not part of the problem match; with more than two groups all pairwise rank-sum comparisons are Holm-corrected.")}</p></div>
          <Button tone="primary" className="sa-run-button" disabled={!canCreate || creating} onClick={() => void createAnalysis()} icon={<Sigma size={15} />}>{creating ? tr("Computing and writing the results…") : tr("Run significance analysis")}</Button>
          {!canCreate ? <small className="sa-config__hint">{tr("Choose at least {0} compatible runsets, one common metric and one output format, and enter a valid name and α.", minimum)}</small> : null}
        </div>
      </aside>

      <ConfirmDialog open={clearOpen} title={tr("Clear the SA candidate list?")} message={tr("Only the candidate references prepared for later analyses are removed; the result library, run records and completed SA history are not deleted.")} confirmLabel={tr("Clear candidate list")} onClose={() => setClearOpen(false)} onConfirm={() => void clearImports()} />
      <ConfirmDialog open={quickFilterDuplicateConfirmOpen} title={tr("Automatic selection includes duplicate algorithm configurations?")} message={tr("The result library holds the same algorithm with the same full parameters as the current SA candidates. These duplicates are not added by default; if you do want to compare different sources or runsets, confirm to add all compatible items at once.")} confirmLabel={tr("Add compatible items anyway")} busy={quickFiltering} onClose={() => { if (!quickFiltering) setQuickFilterDuplicateConfirmOpen(false); }} onConfirm={() => void quickFilter(true)} />
    </section>
  );
}

function AnalysisHistory({ analyses, selectedAnalysisId, onSelectAnalysis, onChanged }: { analyses: SAAnalysis[]; selectedAnalysisId: string | null; onSelectAnalysis: (id: string) => void; onChanged: () => void }) {
  const [detail, setDetail] = useState<SAAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ analysisId: string; value: Error } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Pick<SAAnalysis, "id" | "name"> | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (!selectedAnalysisId) {
      setDetail(null);
      setError(null);
      setLoading(false);
      return;
    }
    let active = true;
    setDetail(null);
    setError(null);
    setLoading(true);
    void api.sa.getAnalysis(selectedAnalysisId).then((item) => { if (active) { setDetail(item); setError(null); } }).catch((nextError) => { if (active) setError({ analysisId: selectedAnalysisId, value: nextError instanceof Error ? nextError : new Error(tr("Could not read the analysis details")) }); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [selectedAnalysisId]);

  const remove = async () => {
    const target = deleteTarget;
    if (!target) return;
    setDeleting(true);
    try { await api.sa.removeAnalysis(target.id); setDeleteTarget(null); setDetail((current) => current?.id === target.id ? null : current); onChanged(); toast.success(tr("SA analysis and its output files deleted")); }
    catch (nextError) { toast.error(tr("Deletion failed"), { description: nextError instanceof Error ? nextError.message : tr("Unknown error") }); }
    finally { setDeleting(false); }
  };

  const visibleDetail = detail?.id === selectedAnalysisId ? detail : null;
  const visibleError = error?.analysisId === selectedAnalysisId ? error.value : null;

  if (!analyses.length) return <EmptyState title={tr("No SA history yet")} description={tr("Run an analysis in the SA workbench after choosing compatible runsets and metrics; the result tables and output paths are kept here.")} />;

  return (
    <section className="analysis-history">
      <aside className="analysis-history__list panel">
        <header className="panel__header"><h2>{tr("Analysis records")}</h2><span className="panel-count">{analyses.length}</span></header>
        <div>{analyses.map((analysis) => <button key={analysis.id} className={selectedAnalysisId === analysis.id ? "is-active" : ""} onClick={() => onSelectAnalysis(analysis.id)}><span><strong>{analysis.name}</strong><Badge tone={analysis.status === "completed" ? "success" : analysis.status === "failed" ? "danger" : analysis.status === "running" ? "info" : "neutral"}>{analysis.status}</Badge></span><p>{methodLabels[analysis.method] ?? analysis.method}</p><small>{analysis.metric_names.join(tr(", ")) || tr("No metrics yet")} · {analysis.source_imports.length} runsets · {formatDateTime(analysis.completed_at ?? analysis.created_at)}</small></button>)}</div>
      </aside>

      <div className="analysis-detail panel">
        {loading || (selectedAnalysisId && !visibleDetail && !visibleError) ? <LoadingBlock label={tr("Reading the analysis results…")} /> : visibleError ? <EmptyState title={tr("Could not read the analysis details")} description={visibleError.message} /> : visibleDetail ? <AnalysisDetail key={visibleDetail.id} analysis={visibleDetail} onDelete={() => setDeleteTarget({ id: visibleDetail.id, name: visibleDetail.name })} /> : <EmptyState title={tr("Choose an analysis record")} description={tr("The evidence of an analysis is shown in three views: statistical results, source tasks and runsets, and statistical details.")} />}
      </div>
      <ConfirmDialog open={Boolean(deleteTarget)} title={tr("Delete this SA analysis?")} message={tr("The database record of “{0}” and the SQLite, CSV, XLSX and LaTeX output folders managed by Relay are deleted permanently; the result library, run records and SA candidate references are not deleted.", deleteTarget?.name ?? tr("this analysis"))} confirmLabel={tr("Delete analysis permanently")} destructive busy={deleting} onClose={() => { if (!deleting) setDeleteTarget(null); }} onConfirm={() => void remove()} />
    </section>
  );
}

function AnalysisDetail({ analysis, onDelete }: { analysis: SAAnalysis; onDelete: () => void }) {
  const payload = asResultPayload(analysis);
  const [tab, setTab] = useState<AnalysisDetailTab>("results");
  const [activeSource, setActiveSource] = useState<SAAnalysis["source_imports"][number] | null>(null);
  const orderedSources = analysis.source_imports;
  const storedLabelDimension = payload?.label_dimension ?? orderedSources.find((source) => source.label_dimension)?.label_dimension;
  const hasStoredLabelDimension = Boolean(storedLabelDimension);
  const storedParameter = storedLabelDimension?.kind === "algorithm_parameter" ? storedLabelDimension.parameter ?? undefined : undefined;
  const storedMode = storedParameter ? `algorithmParam:${storedParameter}` as ComparisonLabelMode : "algorithm";
  const labelPlan = useMemo(() => comparisonLabelPlan(orderedSources, storedMode), [orderedSources, storedMode]);
  const sourceLabels = useMemo(() => {
    if (!hasStoredLabelDimension) return new Map<string, string>();
    return new Map(orderedSources.map((source, index) => [
      source.import_id,
      source.display_label ?? payload?.display_labels?.[index] ?? labelPlan.labels[index],
    ]));
  }, [hasStoredLabelDimension, labelPlan.labels, orderedSources, payload?.display_labels]);
  const labelContext = !hasStoredLabelDimension ? tr("Original column headers of the analysis") : storedParameter ? tr("Grouped by parameter: {0}", storedParameter) : tr("Grouped by algorithm");

  const sourceForTableColumn = (columnIndex: number) => {
    if (columnIndex === 0) return null;
    return orderedSources[columnIndex - 1] ?? null;
  };

  const displayColumn = (column: string, columnIndex: number) => {
    const source = sourceForTableColumn(columnIndex);
    return source && hasStoredLabelDimension ? sourceLabels.get(source.import_id) ?? column : column;
  };

  return (
    <>
      <header className="analysis-detail__header"><div><h2>{analysis.name}</h2><p>{analysis.id}</p></div><Button tone="danger" onClick={onDelete} icon={<Trash2 size={14} />}>{tr("Delete analysis")}</Button></header>
      <div className="analysis-detail__body">
        <section className="analysis-detail-facts"><span><small>{tr("Status")}</small><strong>{analysis.status}</strong></span><span><small>{tr("Method")}</small><strong>{analysis.method}</strong></span><span><small>α</small><strong>{analysis.alpha}</strong></span><span><small>{tr("Metric")}</small><strong>{analysis.metric_names.length}</strong></span><span><small>Runsets</small><strong>{analysis.source_imports.length}</strong></span></section>
        {analysis.error_message ? <Notice tone="danger"><strong>{tr("Analysis failed")}</strong><p>{analysis.error_message}</p></Notice> : null}

        <Tabs
          variant="underline"
          className="analysis-detail-tabs"
          ariaLabel={tr("Analysis detail views")}
          value={tab}
          onChange={setTab}
          items={[
            { value: "results", label: tr("Statistical results"), count: payload?.tables.length ?? 0, icon: <BarChart3 size={14} /> },
            { value: "sources", label: tr("Source tasks and runsets"), count: analysis.source_imports.length, icon: <Database size={14} /> },
            { value: "technical", label: tr("Statistical details"), count: payload?.analyses.length ?? 0, icon: <Sigma size={14} /> },
          ]}
        />

        {tab === "results" ? <div className="analysis-detail-tab-panel">
          {payload?.tables.length ? payload.tables.map((table) => <section key={table.metric_name} className="analysis-result-section"><header><h3>{tr("{0} statistical results{1}", table.metric_name, storedParameter ? ` · ${storedParameter}` : "")}</h3><span>{payload.method} · α={payload.alpha} · {labelContext}</span></header><div className="result-table-wrap"><table className="result-data-table analysis-paper-table"><thead><tr>{table.columns.map((column, columnIndex) => { const source = sourceForTableColumn(columnIndex); const label = displayColumn(column, columnIndex); return <th key={`${columnIndex}-${column}`}>{source ? <button type="button" title={tr("View the runset details of {0}", label)} onClick={() => setActiveSource(source)}>{label}<ChevronRight size={12} /></button> : label}</th>; })}</tr></thead><tbody>{table.rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{String(cell ?? "—")}</td>)}</tr>)}</tbody></table></div></section>) : <EmptyState title={tr("No statistical results")} description={analysis.status === "failed" ? tr("The analysis failed; the error message is kept above.") : tr("This analysis produced no metric table to show.")} />}
          <section className="analysis-result-section"><header><h3>{tr("Output files")}</h3><span>{tr("{0} files", Object.keys(analysis.output_files).length)}</span></header>{Object.keys(analysis.output_files).length ? <div className="analysis-output-list">{Object.entries(analysis.output_files).map(([format, path]) => <div key={format}><strong>{format}</strong><code>{path}</code></div>)}</div> : <p className="muted-copy">{tr("No files were written; the error message is kept above.")}</p>}</section>
        </div> : null}

        {tab === "sources" ? <div className="analysis-detail-tab-panel"><section className="analysis-result-section analysis-source-section"><header><h3>{tr("Source tasks and runsets")}</h3><span>{tr("Click any item to view its frozen parameters, summary metrics and on-demand data")}</span></header><div className="analysis-source-list">{orderedSources.map((source) => <button type="button" key={source.import_id} onClick={() => setActiveSource(source)}><div><strong>{source.task_name}</strong><p>{source.label} · {source.algorithm_name}{storedParameter ? ` · ${storedParameter}=${sourceLabels.get(source.import_id) ?? "—"}` : ""}</p></div><span>{source.problem_name} · {source.completed_runs}/{source.expected_runs ?? source.completed_runs} runs</span><code title={source.runset_uid}>{source.runset_uid.slice(0, 10)}</code><ChevronRight size={14} /></button>)}</div></section></div> : null}

        {tab === "technical" ? <div className="analysis-detail-tab-panel">{payload?.analyses.length ? payload.analyses.map((item, index) => <TechnicalAnalysis key={String(item.metric_name ?? index)} item={item} alpha={analysis.alpha} sourceLabels={sourceLabels} />) : <EmptyState title={tr("No statistical details")} description={tr("This analysis has no overall test, ranking or pairwise comparison records.")} />}</div> : null}
      </div>
      <AnalysisRunsetDrawer source={activeSource} onClose={() => setActiveSource(null)} />
    </>
  );
}

function AnalysisRunsetDrawer({ source, onClose }: { source: SAAnalysis["source_imports"][number] | null; onClose: () => void }) {
  const [runset, setRunset] = useState<ResultRunset | null>(source ? frozenSourceRunset(source) : null);
  const [loading, setLoading] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);

  useEffect(() => {
    if (!source) {
      setRunset(null);
      setWarning(null);
      return;
    }
    let active = true;
    setRunset(frozenSourceRunset(source));
    setWarning(null);
    setLoading(true);
    void api.results.preview(source.result_id ?? source.run_id).then((preview) => {
      if (!active) return;
      const liveRunset = preview.runsets.find((item) => item.runset_uid === source.runset_uid);
      if (liveRunset) setRunset(liveRunset);
      else setWarning(tr("The source result still exists but no longer contains this runset; the parameters and summary frozen at analysis time are shown."));
    }).catch(() => {
      if (active) setWarning(tr("The source result is unavailable; the content below comes from the snapshot frozen in the analysis record, and raw samples and trends may no longer be readable."));
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [source]);

  return (
    <ResultDrawer
      open={Boolean(source)}
      title={source ? `${source.label} · ${source.algorithm_name}` : tr("Runset details")}
      description={source ? `${source.problem_name} · ${source.completed_runs}/${source.expected_runs ?? source.completed_runs} runs · ${source.runset_uid}` : undefined}
      onClose={onClose}
    >
      {loading ? <p className="analysis-source-loading"><RefreshCw size={13} className="is-spinning" />{tr("Checking the latest read-only preview of the source runset…")}</p> : null}
      {warning ? <Notice tone="warning"><strong>{tr("Using the frozen source")}</strong><p>{warning}</p></Notice> : null}
      {source && runset ? <RunsetDetail key={`${source.result_id ?? source.run_id}:${source.runset_uid}:${runset.scalar_metrics.length}:${runset.trend_metrics.map((item) => item.name).join(",")}`} runId={source.result_id ?? source.run_id} runset={runset} /> : null}
    </ResultDrawer>
  );
}

function frozenSourceRunset(source: SAAnalysis["source_imports"][number]): ResultRunset {
  const expectedRuns = source.expected_runs ?? source.completed_runs;
  return {
    runset_id: 0,
    runset_uid: source.runset_uid,
    task_id: 0,
    task_uid: source.import_id,
    label: source.label,
    algorithm_name: source.algorithm_name,
    problem_name: source.problem_name,
    algorithm_params: source.algorithm_params,
    problem_params: source.problem_params,
    expected_runs: expectedRuns,
    completed_runs: source.completed_runs,
    status: source.status ?? (source.completed_runs === expectedRuns ? "completed" : "unknown"),
    task_config_hash: "",
    scalar_metrics: source.scalar_metrics ?? [],
    trend_metrics: source.trend_metrics ?? [],
    pairing_signature: "",
    sa_import_id: source.import_id,
  };
}

function TechnicalAnalysis({ item, alpha, sourceLabels }: { item: Record<string, unknown>; alpha: number; sourceLabels: Map<string, string> }) {
  const metricName = String(item.metric_name ?? "metric");
  const rankRows = Array.isArray(item.rank_rows) ? item.rank_rows as Array<Record<string, unknown>> : [];
  const pairwiseRows = Array.isArray(item.rows) ? item.rows as Array<Record<string, unknown>> : Array.isArray(item.pairwise_rows) ? item.pairwise_rows as Array<Record<string, unknown>> : [];
  const statistic = typeof item.statistic === "number" ? item.statistic : null;
  const pValue = typeof item.p_value === "number" ? item.p_value : null;
  return (
    <section className="analysis-result-section">
      <header><h3>{tr("{0} · statistical details", metricName)}</h3><span>{String(item.direction ?? "—")}</span></header>
      {statistic != null || pValue != null ? <div className="omnibus-strip"><span>Statistic <strong>{formatMetricValue(statistic)}</strong></span><span>p-value <strong className={pValue != null && pValue < alpha ? "is-significant" : ""}>{formatMetricValue(pValue)}</strong></span>{typeof item.kendalls_w === "number" ? <span>Kendall's W <strong>{formatMetricValue(item.kendalls_w)}</strong></span> : null}</div> : null}
      {rankRows.length ? <div className="result-table-wrap"><table className="result-data-table"><thead><tr><th>Runset</th><th>{tr("Algorithm")}</th><th>Mean</th><th>Median</th><th>{tr("Mean rank")}</th><th>{tr("Group")}</th></tr></thead><tbody>{rankRows.map((row, index) => { const id = String(row.id ?? index); return <tr key={id}><td>{sourceLabels.get(id) ?? String(row.label ?? row.id ?? "—")}</td><td>{String(row.algorithm ?? "—")}</td><td>{formatMetricValue(typeof row.mean === "number" ? row.mean : null)}</td><td>{formatMetricValue(typeof row.median === "number" ? row.median : null)}</td><td>{formatMetricValue(typeof row.average_rank === "number" ? row.average_rank : null)}</td><td><strong>{String(row.posthoc_group ?? "—")}</strong></td></tr>; })}</tbody></table></div> : null}
      {pairwiseRows.length ? <div className="result-table-wrap"><table className="result-data-table pairwise-table"><thead><tr><th>{tr("Left")}</th><th>{tr("Right")}</th><th>p</th><th>{tr("Adjusted p")}</th><th>{tr("Significant")}</th><th>{tr("Winner")}</th><th>{tr("Effect size")}</th></tr></thead><tbody>{pairwiseRows.map((row, index) => { const rawP = typeof row.p_value === "number" ? row.p_value : null; const adjustedP = typeof row.adjusted_p_value === "number" ? row.adjusted_p_value : rawP; const significant = adjustedP != null && adjustedP < alpha; const winner = row.winner_id == null ? tr("Tie") : row.winner_id === row.left_id ? tr("Left") : row.winner_id === row.right_id ? tr("Right") : tr("Unknown"); const leftId = String(row.left_id ?? ""); const rightId = String(row.right_id ?? ""); return <tr key={index}><td>{sourceLabels.get(leftId) ?? String(row.left_label ?? row.left_id ?? "—")}</td><td>{sourceLabels.get(rightId) ?? String(row.right_label ?? row.right_id ?? "—")}</td><td>{formatMetricValue(rawP)}</td><td className={significant ? "is-significant" : ""}>{formatMetricValue(adjustedP)}</td><td><Badge tone={significant ? "danger" : "neutral"}>{significant ? tr("Significant") : tr("Not significant")}</Badge></td><td>{winner}</td><td>{typeof row.effect_size === "number" ? `${formatMetricValue(row.effect_size)} · ${String(row.effect_size_name ?? "")}` : "—"}</td></tr>; })}</tbody></table></div> : null}
    </section>
  );
}
