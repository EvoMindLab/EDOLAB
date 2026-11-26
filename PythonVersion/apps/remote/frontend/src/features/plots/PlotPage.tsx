import { useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  Box,
  ChevronDown,
  Download,
  Image as ImageIcon,
  LineChart,
  ListPlus,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { Badge, Button, EmptyState, LoadingBlock, Notice, PageHeader, Tabs } from "../../components/ui";
import { comparisonLabelPlan, type ComparisonLabelMode } from "../../lib/comparisonLabels";
import "./plots.css";
import { tr, LOCALE } from "../../lib/i18n";

export type PlotKind = "trend" | "boxplot";
export type PlotFormat = "png" | "pdf" | "svg";
export type PlotLegendMode = ComparisonLabelMode;

export interface PlotSource {
  id: string;
  run_id: string;
  runset_uid: string;
  label: string;
  algorithm_name: string;
  problem_name: string;
  scalar_metrics: string[];
  trend_metrics: string[];
  trend_metric_axes?: Record<string, string>;
  algorithm_params?: Record<string, unknown>;
  task_name?: string;
  completed_runs?: number;
  expected_runs?: number;
  status?: string;
  available?: boolean;
  detail?: string;
}

export interface PlotTrendSeries {
  label: string;
  axis: string;
  key: Array<number | string>;
  mean: number[];
  stderr?: number[];
  count?: number[];
}

export interface PlotTrendSeriesRequest {
  xMin?: number;
  xMax?: number;
  maxPoints?: number;
}

export interface PlotBoxplotSeries {
  label: string;
  values: number[];
}

interface PlotInputBase {
  metric_name: string;
  title?: string;
}

export interface PlotTrendPreviewInput extends PlotInputBase {
  kind: "trend";
  series: PlotTrendSeries[];
  x_min?: number;
  x_max?: number;
  legend_position?: string;
  y_scale?: "auto" | "linear" | "log";
}

export interface PlotBoxplotPreviewInput extends PlotInputBase {
  kind: "boxplot";
  series: PlotBoxplotSeries[];
}

export type PlotPreviewInput = PlotTrendPreviewInput | PlotBoxplotPreviewInput;
export type PlotTrendInput = PlotTrendPreviewInput & { formats: PlotFormat[] };
export type PlotBoxplotInput = PlotBoxplotPreviewInput & { formats: PlotFormat[] };
export type PlotCreateInput = PlotTrendInput | PlotBoxplotInput;

export interface PlotArtifact {
  id: string;
  storage_key: string;
  filename: string;
  format: PlotFormat;
  media_type: string;
  size_bytes: number;
  sha256: string;
  url?: string;
  created_at?: string;
}

export interface PlotCompletedResult {
  kind: "trend_plot" | "boxplot";
  metric_name: string;
  series_count: number;
  point_count: number;
  axis?: string;
  artifacts: Partial<Record<PlotFormat, PlotArtifact>>;
  preview_artifact?: PlotArtifact | null;
}

export interface PlotPreviewResult {
  kind: "trend_plot" | "boxplot";
  metric_name: string;
  series_count: number;
  point_count: number;
  axis?: string;
  image: Blob;
}

export interface PlotApi {
  listSources: () => Promise<PlotSource[]>;
  loadTrendSeries: (sourceId: string, metricName: string, options?: PlotTrendSeriesRequest) => Promise<Omit<PlotTrendSeries, "label">>;
  loadBoxplotSeries: (sourceId: string, metricName: string) => Promise<Omit<PlotBoxplotSeries, "label">>;
  createPreview: (input: PlotPreviewInput) => Promise<PlotPreviewResult>;
  createPlot: (input: PlotCreateInput) => Promise<PlotCompletedResult>;
  artifactUrl?: (artifact: PlotArtifact) => string;
}

export interface PlotPageProps {
  api: PlotApi;
  refreshNonce?: number;
  sourceRefreshToken?: number;
  active?: boolean;
  embedded?: boolean;
  scopeRunId?: string;
  scopeTaskName?: string;
  initialSourceId?: string;
  selectionSignal?: number;
}

interface DisplayedPlotPreview extends Omit<PlotPreviewResult, "image"> {
  imageUrl: string;
}

const formatOptions: Array<{ value: PlotFormat; label: string }> = [
  { value: "png", label: "PNG" },
  { value: "pdf", label: "PDF" },
  { value: "svg", label: "SVG" },
];

const TOTAL_TREND_POINT_BUDGET = 200_000;
const MAX_TREND_POINTS_PER_SERIES = 20_000;

export const allocateTrendPointBudgets = (seriesCount: number): number[] => {
  if (!Number.isInteger(seriesCount) || seriesCount <= 0) return [];
  if (seriesCount > TOTAL_TREND_POINT_BUDGET) throw new Error(tr("Too many trend series are selected to give each a useful sampling budget."));
  const base = Math.floor(TOTAL_TREND_POINT_BUDGET / seriesCount);
  const remainder = TOTAL_TREND_POINT_BUDGET % seriesCount;
  return Array.from(
    { length: seriesCount },
    (_, index) => Math.min(MAX_TREND_POINTS_PER_SERIES, base + (index < remainder ? 1 : 0)),
  );
};

const legendPositionOptions = [
  ["outside_right", tr("Outside, right")],
  ["best", tr("Inside, automatic")],
  ["upper_left", tr("Upper left")],
  ["upper_right", tr("Upper right")],
  ["lower_left", tr("Lower left")],
  ["lower_right", tr("Lower right")],
] as const;

export function PlotPage({
  api,
  refreshNonce = 0,
  sourceRefreshToken = 0,
  active = true,
  embedded = false,
  scopeRunId,
  scopeTaskName,
  initialSourceId,
  selectionSignal = 0,
}: PlotPageProps) {
  const [sources, setSources] = useState<PlotSource[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<PlotKind>("trend");
  const [metricName, setMetricName] = useState("");
  const [legendMode, setLegendMode] = useState<PlotLegendMode>("algorithm");
  const [legendPosition, setLegendPosition] = useState("outside_right");
  const [xMin, setXMin] = useState("");
  const [xMax, setXMax] = useState("");
  const [yScale, setYScale] = useState<"auto" | "linear" | "log">("auto");
  const [title, setTitle] = useState("");
  const [formats, setFormats] = useState<Set<PlotFormat>>(new Set(["png", "pdf", "svg"]));
  const [previewResult, setPreviewResult] = useState<DisplayedPlotPreview | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [exportResult, setExportResult] = useState<PlotCompletedResult | null>(null);
  const [otherTasksOpen, setOtherTasksOpen] = useState(false);
  const [metricNotice, setMetricNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [busy, setBusy] = useState<"preview" | "export" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activePlotRequest = useRef(0);
  const previewUrlRef = useRef<string | null>(null);

  const replacePreview = (next: PlotPreviewResult | null) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = next ? URL.createObjectURL(next.image) : null;
    if (!next) setPreviewOpen(false);
    setPreviewResult(next && previewUrlRef.current ? {
      kind: next.kind,
      metric_name: next.metric_name,
      series_count: next.series_count,
      point_count: next.point_count,
      axis: next.axis,
      imageUrl: previewUrlRef.current,
    } : null);
  };

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  const loadSources = async (quiet = false) => {
    if (!quiet) setRefreshing(true);
    try {
      const nextSources = await api.listSources();
      setSources(nextSources);
      setSelectedIds((current) => new Set([...current].filter((id) => nextSources.some((item) => item.id === id && sourceAvailable(item)))));
      setError(null);
    } catch (nextError) {
      setError(errorMessage(nextError, tr("Could not read the plot sources")));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    if (!active) return;
    let requestActive = true;
    setLoading(true);
    api.listSources()
      .then((nextSources) => {
        if (!requestActive) return;
        setSources(nextSources);
        setSelectedIds((current) => new Set([...current].filter((id) => nextSources.some((item) => item.id === id && sourceAvailable(item)))));
        setError(null);
      })
      .catch((nextError: unknown) => {
        if (requestActive) setError(errorMessage(nextError, tr("Could not read the plot sources")));
      })
      .finally(() => { if (requestActive) setLoading(false); });
    return () => { requestActive = false; };
  }, [active, api, refreshNonce, sourceRefreshToken]);

  const currentTaskSources = useMemo(
    () => scopeRunId ? sources.filter((source) => source.run_id === scopeRunId) : sources,
    [scopeRunId, sources],
  );
  const otherTaskSources = useMemo(
    () => scopeRunId ? sources.filter((source) => source.run_id !== scopeRunId) : [],
    [scopeRunId, sources],
  );
  const filteredOtherTaskSources = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return otherTaskSources.filter((source) => !normalized || [
      source.task_name,
      source.label,
      source.algorithm_name,
      source.problem_name,
      source.runset_uid,
    ].join(" ").toLowerCase().includes(normalized));
  }, [otherTaskSources, query]);

  const selectedSources = useMemo(
    () => sources.filter((source) => selectedIds.has(source.id) && sourceAvailable(source)),
    [selectedIds, sources],
  );
  const metricAxes = useMemo(() => commonMetrics(selectedSources, kind), [kind, selectedSources]);
  const metricNames = useMemo(() => [...metricAxes.keys()].sort(), [metricAxes]);
  const labelPlan = useMemo(() => comparisonLabelPlan(selectedSources, legendMode), [legendMode, selectedSources]);
  const axis = kind === "trend" && metricName ? metricAxes.get(metricName) : undefined;
  const availableCurrentTask = currentTaskSources.filter(sourceAvailable);
  const allCurrentTaskSelected = availableCurrentTask.length > 0 && availableCurrentTask.every((source) => selectedIds.has(source.id));
  const groupedOtherTaskSources = useMemo(() => groupSourcesByTask(filteredOtherTaskSources), [filteredOtherTaskSources]);

  useEffect(() => {
    if (!initialSourceId || !selectionSignal) return;
    const source = sources.find((item) => item.id === initialSourceId);
    if (!source || !sourceAvailable(source)) return;
    setSelectedIds((current) => new Set(current).add(source.id));
  }, [initialSourceId, selectionSignal, sources]);

  useEffect(() => {
    if (metricNames.includes(metricName)) return;
    setMetricName(metricNames[0] ?? "");
  }, [metricName, metricNames]);

  useEffect(() => {
    if (!selectedSources.length) {
      setMetricNotice(null);
      return;
    }
    const trendMetrics = commonMetrics(selectedSources, "trend");
    const summaryMetrics = commonMetrics(selectedSources, "boxplot");
    if (trendMetrics.size === 0 && summaryMetrics.size > 0) {
      const reason = noCommonMetricReason(selectedSources, "trend");
      if (kind === "trend") {
        setMetricNotice(tr("{0} switched to the summary box plot automatically; {1} common summary metrics are available.", reason, summaryMetrics.size));
        setKind("boxplot");
        setMetricName([...summaryMetrics.keys()].sort()[0] ?? "");
        replacePreview(null);
      } else {
        setMetricNotice(tr("{0} uses the summary box plot; {1} common summary metrics are available.", reason, summaryMetrics.size));
      }
      return;
    }
    if (kind === "trend" && metricAxes.size === 0) {
      setMetricNotice(noCommonMetricReason(selectedSources, "trend"));
      return;
    }
    setMetricNotice(null);
  }, [kind, metricAxes, selectedSources]);

  useEffect(() => {
    if (labelPlan.mode !== legendMode) setLegendMode(labelPlan.mode);
  }, [labelPlan.mode, legendMode]);

  useEffect(() => {
    activePlotRequest.current += 1;
    setBusy(null);
    replacePreview(null);
  }, [kind, legendMode, legendPosition, metricName, selectedSources, title, xMax, xMin, yScale]);

  const toggleSource = (source: PlotSource) => {
    if (!sourceAvailable(source)) return;
    setMetricNotice(null);
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(source.id)) next.delete(source.id); else next.add(source.id);
      return next;
    });
  };

  const toggleCurrentTask = () => {
    setMetricNotice(null);
    setSelectedIds((current) => {
      const next = new Set(current);
      for (const source of availableCurrentTask) {
        if (allCurrentTaskSelected) next.delete(source.id); else next.add(source.id);
      }
      return next;
    });
  };

  const toggleFormat = (format: PlotFormat) => {
    setFormats((current) => {
      const next = new Set(current);
      if (next.has(format)) next.delete(format); else next.add(format);
      return next;
    });
  };

  const createPlot = async (mode: "preview" | "export") => {
    setError(null);
    if (!selectedSources.length) {
      setError(tr("Choose at least one complete result source."));
      return;
    }
    if (!metricName || !metricAxes.has(metricName)) {
      setError(noCommonMetricReason(selectedSources, kind));
      return;
    }
    if (mode === "export" && formats.size === 0) {
      setError(tr("Choose at least one export format."));
      return;
    }

    let xMinimum: number | undefined;
    let xMaximum: number | undefined;
    try {
      xMinimum = kind === "trend" ? optionalFiniteNumber(xMin, tr("x-axis start")) : undefined;
      xMaximum = kind === "trend" ? optionalFiniteNumber(xMax, tr("x-axis end")) : undefined;
      if (xMinimum !== undefined && xMaximum !== undefined && xMinimum > xMaximum) {
        throw new Error(tr("The x-axis start cannot be greater than the end."));
      }
    } catch (nextError) {
      setError(errorMessage(nextError, tr("Invalid x-axis range")));
      return;
    }

    const requestId = activePlotRequest.current + 1;
    activePlotRequest.current = requestId;
    setBusy(mode);
    try {
      const activeLabelPlan = comparisonLabelPlan(selectedSources, legendMode);
      const labels = activeLabelPlan.labels;
      const resolvedTitle = title.trim() || (activeLabelPlan.parameter ? `${metricName} · ${activeLabelPlan.title}` : undefined);
      let input: PlotPreviewInput;
      if (kind === "trend") {
        const pointBudgets = allocateTrendPointBudgets(selectedSources.length);
        const loadedSeries = await Promise.all(selectedSources.map(async (source, index) => {
          try {
            return {
              ...await api.loadTrendSeries(source.id, metricName, {
                xMin: xMinimum,
                xMax: xMaximum,
                maxPoints: pointBudgets[index],
              }),
              label: labels[index],
            };
          } catch (nextError) {
            if (errorCode(nextError) === "plot_trend_range_empty") return null;
            throw new Error(tr("{0}: {1}", labels[index], errorMessage(nextError, tr("Could not read the trend data"))));
          }
        }));
        const series = loadedSeries.filter((item): item is PlotTrendSeries => item !== null);
        const skippedLabels = loadedSeries.flatMap((item, index) => item === null ? [labels[index]] : []);
        if (!series.length) {
          throw new Error(tr("None of the selected trend series has data in the given x-axis range; adjust the range and try again."));
        }
        setMetricNotice(skippedLabels.length
          ? tr("{0} series have no data in the given x-axis range and were skipped: {1}{2}.", skippedLabels.length, skippedLabels.slice(0, 3).join(tr(", ")), skippedLabels.length > 3 ? tr(" and others") : "")
          : null);
        input = {
          kind,
          metric_name: metricName,
          series,
          title: resolvedTitle,
          x_min: xMinimum,
          x_max: xMaximum,
          legend_position: legendPosition,
          y_scale: yScale,
        };
      } else {
        const series = await Promise.all(selectedSources.map(async (source, index) => ({
          ...await api.loadBoxplotSeries(source.id, metricName),
          label: labels[index],
        })));
        input = {
          kind,
          metric_name: metricName,
          series,
          title: resolvedTitle,
        };
      }
      if (mode === "preview") {
        const response = await api.createPreview(input);
        if (activePlotRequest.current === requestId) {
          replacePreview(response);
          setPreviewOpen(true);
        }
      } else {
        const response = await api.createPlot({ ...input, formats: [...formats] } as PlotCreateInput);
        if (activePlotRequest.current === requestId) setExportResult(response);
      }
    } catch (nextError) {
      if (activePlotRequest.current === requestId) setError(errorMessage(nextError, tr("Plot failed")));
    } finally {
      if (activePlotRequest.current === requestId) setBusy(null);
    }
  };

  if (loading) return <LoadingBlock label={tr("Reading the local results that can be plotted…")} />;

  return (
    <div className={embedded ? "plot-workspace plot-workspace--embedded" : "page plot-workspace"}>
      {!embedded ? (
        <PageHeader
          title={tr("Result plots")}
          description={tr("Trend plots and summary box plots drawn on the fly from imported runsets")}
          actions={<Button onClick={() => void loadSources()} disabled={refreshing} icon={<RefreshCw size={15} className={refreshing ? "is-spinning" : ""} />}>{refreshing ? tr("Refreshing…") : tr("Refresh plot sources")}</Button>}
        />
      ) : null}

      {error ? <Notice tone="danger"><strong>{tr("Cannot plot")}</strong><p>{error}</p></Notice> : null}
      {metricNotice ? <Notice><strong>{tr("Metric descriptions")}</strong><p>{metricNotice}</p></Notice> : null}

      <div className="plot-layout">
        <section className="panel plot-source-panel" aria-label={tr("Plot sources")}>
          <header className="panel__header plot-source-header">
            <h2>{tr("Result series")}</h2>
            <div><span className="plot-count">{tr("{0} / {1} selected", selectedSources.length, sources.length)}</span><Button tone="quiet" onClick={() => void loadSources()} disabled={refreshing} icon={<RefreshCw size={13} className={refreshing ? "is-spinning" : ""} />}>{refreshing ? tr("Refreshing") : tr("Refresh")}</Button></div>
          </header>
          <div className="plot-source-toolbar">
            <span className="plot-source-scope">{tr("Current imported result · {0}", scopeTaskName || scopeRunId || tr("All"))}</span>
            <Button tone="quiet" onClick={toggleCurrentTask} disabled={!availableCurrentTask.length}>{allCurrentTaskSelected ? tr("Deselect this task") : tr("Select this task")}</Button>
          </div>
          <div className="plot-source-table-wrap" role="region" aria-label={tr("Available runsets")} tabIndex={0}>
            {currentTaskSources.length ? (
              <table className="plot-source-table">
                <thead><tr><th aria-label={tr("Select")} /><th>Runset</th><th>{tr("Algorithm")}</th><th>{tr("Problem")}</th><th>{tr("Metric")}</th></tr></thead>
                <tbody>{currentTaskSources.map((source) => {
                  const available = sourceAvailable(source);
                  return (
                    <tr key={source.id} className={selectedIds.has(source.id) ? "is-selected" : ""} data-disabled={!available || undefined} onClick={() => toggleSource(source)}>
                      <td><input type="checkbox" aria-label={tr("Select {0}", source.label)} checked={selectedIds.has(source.id)} disabled={!available} onChange={() => toggleSource(source)} onClick={(event) => event.stopPropagation()} /></td>
                      <td title={[source.task_name, source.label, source.runset_uid, source.detail].filter(Boolean).join(" · ")}><strong>{source.label || tr("Unnamed runset")}</strong><small>{shortId(source.runset_uid)}{available ? "" : tr(" · unavailable")}</small></td>
                      <td title={source.algorithm_name}>{source.algorithm_name}</td>
                      <td title={source.problem_name}>{source.problem_name}</td>
                      <td><strong>{source.trend_metrics.length}</strong><small>{tr("Trend")}</small><strong>{source.scalar_metrics.length}</strong><small>{tr("Summary")}</small></td>
                    </tr>
                  );
                })}</tbody>
              </table>
            ) : <EmptyState title={tr("This imported result has no runset that can be plotted")} description={tr("Only runsets imported into the result library and fully validated can be plotted.")} />}
          </div>
          {scopeRunId && otherTaskSources.length ? <div className="plot-other-sources">
            <button type="button" className="plot-other-sources__toggle" aria-expanded={otherTasksOpen} onClick={() => setOtherTasksOpen((current) => !current)}><ListPlus size={14} /><span>{tr("Add series from other imported results")}</span><small>{tr("{0} runsets", otherTaskSources.length)}</small><ChevronDown size={14} /></button>
            {otherTasksOpen ? <div className="plot-other-sources__body"><label className="plot-search"><Search size={14} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("Search imported results, algorithms, problems or runsets")} /></label>{groupedOtherTaskSources.map((group) => <details key={group.runId}><summary><span><strong>{group.taskName}</strong><small>{tr("{0} runsets", group.sources.length)}</small></span><ChevronDown size={13} /></summary><div>{group.sources.map((source) => <label key={source.id} className={selectedIds.has(source.id) ? "is-selected" : ""}><input type="checkbox" checked={selectedIds.has(source.id)} disabled={!sourceAvailable(source)} onChange={() => toggleSource(source)} /><span><strong>{source.label}</strong><small>{source.algorithm_name} · {source.problem_name}</small></span></label>)}</div></details>)}</div> : null}
          </div> : null}
          {selectedSources.length ? <div className="plot-selected-sources"><strong>{tr("Selected plot series")}</strong><div>{selectedSources.map((source) => <button type="button" key={source.id} onClick={() => toggleSource(source)} title={tr("Remove this series")}><span>{source.task_name} · {source.label} · {source.algorithm_name}</span><X size={12} /></button>)}</div></div> : null}
        </section>

        <section className="panel plot-settings-panel" aria-label={tr("Plot settings")}>
          <header className="panel__header"><h2>{tr("Plot settings")}</h2><Badge tone="success">{tr("Drawn on this computer")}</Badge></header>
          <div className="plot-settings-body">
            <Tabs
              variant="segmented"
              className="ui-tabs--stretch"
              ariaLabel={tr("Plot kind")}
              value={kind}
              onChange={(next) => { setMetricNotice(null); setKind(next); replacePreview(null); }}
              items={[
                { value: "trend", label: tr("Trend plot"), icon: <LineChart size={14} /> },
                { value: "boxplot", label: tr("Summary box plot"), icon: <Box size={14} /> },
              ]}
            />

            <div className="plot-fields">
              <label>
                <span>{tr("Metric")}</span>
                <select value={metricName} onChange={(event) => { setMetricName(event.target.value); replacePreview(null); }} disabled={!metricNames.length}>
                  <option value="">{metricSelectPlaceholder(selectedSources, kind, metricNames)}</option>
                  {metricNames.map((metric) => <option key={metric} value={metric}>{metric}</option>)}
                </select>
                <small>{metricHint(selectedSources, kind, metricNames, axis)}</small>
              </label>
              <label>
                <span>{tr("Legend labels")}</span>
                <select value={labelPlan.mode} disabled={labelPlan.mixedAlgorithms || !labelPlan.parameterModes.length} onChange={(event) => setLegendMode(event.target.value as PlotLegendMode)}>
                  {labelPlan.mixedAlgorithms || !labelPlan.parameterModes.length ? <option value="algorithm">{tr("Algorithm name (automatic)")}</option> : null}
                  {labelPlan.parameterModes.map((option) => <option key={option} value={option}>{tr("Algorithm parameter · {0}", option.slice("algorithmParam:".length))}</option>)}
                </select>
                <small>{labelPlan.mixedAlgorithms ? tr("Several algorithms detected; algorithm names are used.") : labelPlan.parameterModes.length ? tr("Single-algorithm experiment, labelled by the varying parameter {0}; repeated values get a short runset ID.", labelPlan.title) : tr("No algorithm parameter varies; repeated algorithm names get a short runset ID.")}</small>
              </label>
              {kind === "trend" ? (
                <label>
                  <span>{tr("Legend position")}</span>
                  <select value={legendPosition} onChange={(event) => setLegendPosition(event.target.value)}>{legendPositionOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
                  <small>{tr("Outside right suits comparisons with many series.")}</small>
                </label>
              ) : null}
              <label className="plot-field-wide">
                <span>{tr("Title (optional)")}</span>
                <input value={title} maxLength={2048} onChange={(event) => setTitle(event.target.value)} placeholder={`${kind === "trend" ? "Trend" : "Boxplot"} of ${metricName || "metric"}`} />
                <small>{labelPlan.parameter ? tr("When left empty, the title names the parameter {0}.", labelPlan.title) : tr("When left empty, the default title is used.")}</small>
              </label>
              {kind === "trend" ? (
                <>
                  <label>
                    <span>{tr("{0} start", axisRangeLabel(axis))}</span>
                    <input inputMode="decimal" value={xMin} onChange={(event) => setXMin(event.target.value)} placeholder={tr("Automatic")} />
                    <small>{tr("Starts at the first available sample when left empty.")}</small>
                  </label>
                  <label>
                    <span>{tr("{0} end", axisRangeLabel(axis))}</span>
                    <input inputMode="decimal" value={xMax} onChange={(event) => setXMax(event.target.value)} placeholder={tr("Automatic")} />
                    <small>{tr("Trend samples outside the range are not drawn.")}</small>
                  </label>
                  <label>
                    <span>{tr("y-axis scale")}</span>
                    <select value={yScale} onChange={(event) => setYScale(event.target.value as "auto" | "linear" | "log")}>
                      <option value="auto">{tr("Automatic")}</option><option value="linear">{tr("Linear")}</option><option value="log">{tr("Logarithmic")}</option>
                    </select>
                    <small>{tr("Automatic mode chooses by metric and by the valid values.")}</small>
                  </label>
                </>
              ) : null}
            </div>

            <fieldset className="plot-format-field">
              <legend>{tr("Export formats")}</legend>
              {formatOptions.map((option) => <label key={option.value}><input type="checkbox" checked={formats.has(option.value)} onChange={() => toggleFormat(option.value)} /><span>{option.label}</span></label>)}
              <small>{tr("Applies only to “Export figure”. The preview is an in-memory PNG and no file is saved.")}</small>
            </fieldset>

            <div className="plot-actions">
              <Button icon={<ImageIcon size={15} />} disabled={Boolean(busy)} onClick={() => previewResult ? setPreviewOpen(true) : void createPlot("preview")}>{busy === "preview" ? <><RefreshCw size={14} className="is-spinning" />{tr("Generating…")}</> : previewResult ? tr("View preview") : tr("Generate preview")}</Button>
              <Button tone="primary" icon={<Download size={15} />} disabled={Boolean(busy)} onClick={() => void createPlot("export")}>{busy === "export" ? <><RefreshCw size={14} className="is-spinning" />{tr("Exporting…")}</> : tr("Export figure")}</Button>
            </div>
            {busy ? <p className="plot-busy"><RefreshCw size={14} className="is-spinning" />{tr("Reading the metrics and drawing the figure on this computer…")}</p> : null}
            {exportResult ? <PlotExportSummary result={exportResult} api={api} /> : null}
          </div>
        </section>
      </div>

      <PlotPreviewDialog open={previewOpen} preview={previewResult} exported={exportResult} api={api} onClose={() => setPreviewOpen(false)} />
    </div>
  );
}

function plotArtifacts(exported: PlotCompletedResult | null) {
  return formatOptions.flatMap(({ value }) => {
    const artifact = exported?.artifacts[value];
    return artifact ? [artifact] : [];
  });
}

function PlotArtifactLinks({ exported, api }: { exported: PlotCompletedResult | null; api: PlotApi }) {
  const artifacts = plotArtifacts(exported);
  return <div className="plot-artifact-links">{artifacts.length ? artifacts.map((artifact) => {
    const url = api.artifactUrl?.(artifact) ?? artifact.url;
    return url ? <a key={artifact.id} href={url} target="_blank" rel="noreferrer" title={`${artifact.filename} · ${formatBytes(artifact.size_bytes)}`}><Download size={13} />{artifact.format.toUpperCase()}</a> : <span key={artifact.id} title={tr("The API returned no artifact URL")}>{artifact.format.toUpperCase()}</span>;
  }) : <span className="plot-no-artifacts">{tr("Not exported yet")}</span>}</div>;
}

function PlotExportSummary({ result, api }: { result: PlotCompletedResult; api: PlotApi }) {
  return <div className="plot-export-summary"><p><Download size={13} /><span>{tr("{0} · {1} series · {2} points", result.metric_name, result.series_count, result.point_count.toLocaleString(LOCALE))}</span></p><PlotArtifactLinks exported={result} api={api} /></div>;
}

function PlotPreviewDialog({ open, preview, exported, api, onClose }: { open: boolean; preview: DisplayedPlotPreview | null; exported: PlotCompletedResult | null; api: PlotApi; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open || !preview) return null;
  return (
    <div className="plot-preview-modal-layer" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <section className="plot-preview-modal" role="dialog" aria-modal="true" aria-label={tr("Figure preview of {0}", preview.metric_name)}>
        <header><div><h2>{preview.metric_name}</h2><p>{tr("{0} series · {1} data points · in-memory preview", preview.series_count, preview.point_count.toLocaleString(LOCALE))}</p></div><button type="button" className="icon-button" aria-label={tr("Close figure preview")} onClick={onClose}><X size={18} /></button></header>
        <div className="plot-preview-stage"><div className="plot-preview-image-frame"><img src={preview.imageUrl} alt={`${preview.metric_name} ${preview.kind}`} /></div></div>
        <div className="plot-artifact-bar">
          <p><BarChart3 size={14} /><span>{tr("{0}{1} data points", preview.kind === "trend_plot" ? `${axisLabel(preview.axis)} · ` : "", preview.point_count.toLocaleString(LOCALE))}</span></p>
          <PlotArtifactLinks exported={exported} api={api} />
        </div>
      </section>
    </div>
  );
}

interface PlotSourceGroup {
  runId: string;
  taskName: string;
  sources: PlotSource[];
}

function groupSourcesByTask(sources: PlotSource[]): PlotSourceGroup[] {
  const groups = new Map<string, PlotSourceGroup>();
  for (const source of sources) {
    const existing = groups.get(source.run_id);
    if (existing) existing.sources.push(source);
    else groups.set(source.run_id, { runId: source.run_id, taskName: source.task_name || source.run_id, sources: [source] });
  }
  return [...groups.values()].sort((left, right) => left.taskName.localeCompare(right.taskName, LOCALE));
}

function sourceAvailable(source: PlotSource): boolean {
  if (source.available === false) return false;
  if (source.expected_runs !== undefined && source.completed_runs !== undefined && source.completed_runs < source.expected_runs) return false;
  if (!source.status) return true;
  return ["completed", "complete", "succeeded", "finished"].includes(source.status.toLowerCase());
}

function commonMetrics(sources: PlotSource[], kind: PlotKind): Map<string, string | undefined> {
  if (!sources.length) return new Map();
  const names = new Set(metricNamesFor(sources[0], kind));
  for (const source of sources.slice(1)) {
    const available = new Set(metricNamesFor(source, kind));
    for (const metric of names) if (!available.has(metric)) names.delete(metric);
  }
  const result = new Map<string, string | undefined>();
  for (const metric of names) {
    if (kind === "boxplot") {
      result.set(metric, undefined);
      continue;
    }
    const axes = new Set(sources.map((source) => canonicalTrendAxis(source.trend_metric_axes?.[metric])));
    if (axes.size === 1) result.set(metric, [...axes][0]);
  }
  return result;
}

function metricNamesFor(source: PlotSource, kind: PlotKind): string[] {
  const raw = kind === "trend" ? source.trend_metrics : source.scalar_metrics;
  if (!Array.isArray(raw)) return [];
  return [...new Set(raw.filter((item): item is string => typeof item === "string").map((item) => item.trim()).filter(Boolean))];
}

function metricSelectPlaceholder(sources: PlotSource[], kind: PlotKind, commonNames: string[]): string {
  if (!sources.length) return tr("Choose result series first");
  if (commonNames.length) return tr("Choose a metric of type {0}", kind === "trend" ? tr("Trend") : tr("Summary"));
  const metricLists = sources.map((source) => metricNamesFor(source, kind));
  if (metricLists.every((names) => names.length === 0)) {
    return kind === "trend" ? tr("The selected results stored no trend metrics") : tr("The selected results stored no summary metrics");
  }
  return tr("The selected results share no metric of type {0}", kind === "trend" ? tr("Trend") : tr("Summary"));
}

function metricHint(sources: PlotSource[], kind: PlotKind, commonNames: string[], axis?: string): string {
  if (!sources.length) return tr("The metrics are those common to the selected runsets for the current plot kind.");
  if (commonNames.length) {
    if (kind === "trend" && axis) return tr("{0} common trend metrics · x-axis: {1}", commonNames.length, axisLabel(axis));
    return tr("{0} common metrics of type {1}: {2}", commonNames.length, kind === "trend" ? tr("Trend") : tr("Summary"), summarizeNames(commonNames));
  }
  return noCommonMetricReason(sources, kind);
}

function noCommonMetricReason(sources: PlotSource[], kind: PlotKind): string {
  if (!sources.length) return tr("Choose at least one complete result source first.");
  const kindLabel = kind === "trend" ? tr("Trend") : tr("Summary");
  const entries = sources.map((source) => ({ source, names: metricNamesFor(source, kind) }));
  if (entries.every((entry) => entry.names.length === 0)) {
    return kind === "trend"
      ? tr("None of the {0} selected runsets stored trend metrics; this is normal when an experiment writes only summary metrics. To plot trends, the next task must write trend metrics such as current_error.", sources.length)
      : tr("None of the {0} selected runsets stored summary metrics.", sources.length);
  }

  const emptyEntries = entries.filter((entry) => entry.names.length === 0);
  if (emptyEntries.length) {
    return tr("{0} selected runsets stored no metrics of type {1} ({2}), so the selection has no common metric of type {3}.", emptyEntries.length, kindLabel, summarizeSources(emptyEntries.map((entry) => entry.source)), kindLabel);
  }

  const sharedNames = intersectMetricNames(entries.map((entry) => entry.names));
  if (kind === "trend" && sharedNames.length) {
    const incompatibleAxes = sharedNames.flatMap((metric) => {
      const axes = [...new Set(sources.map((source) => canonicalTrendAxis(source.trend_metric_axes?.[metric])))].sort();
      return axes.length > 1 ? [tr("{0} ({1})", metric, axes.map(axisLabel).join(" / "))] : [];
    });
    if (incompatibleAxes.length) {
      return tr("The common trend metrics have different x-axes and cannot be plotted together: {0}.", summarizeNames(incompatibleAxes));
    }
  }

  const details = entries.slice(0, 3).map(({ source, names }) => tr("{0}: [{1}]", sourceLabel(source), summarizeNames(names)));
  const suffix = entries.length > 3 ? tr("; {0} more runsets", entries.length - 3) : "";
  return tr("The selected runsets each stored metrics of type {0}, but no name is shared. {1}{2}.", kindLabel, details.join(tr("; ")), suffix);
}

function intersectMetricNames(metricLists: string[][]): string[] {
  if (!metricLists.length) return [];
  const common = new Set(metricLists[0]);
  for (const names of metricLists.slice(1)) {
    const available = new Set(names);
    for (const name of common) if (!available.has(name)) common.delete(name);
  }
  return [...common].sort();
}

function summarizeNames(names: string[], limit = 5): string {
  if (!names.length) return tr("none");
  const visible = names.slice(0, limit);
  return visible.join(tr(", ")) + (names.length > limit ? tr(" ({0} in total)", names.length) : "");
}

function summarizeSources(sources: PlotSource[], limit = 3): string {
  const visible = sources.slice(0, limit).map(sourceLabel);
  return visible.join(tr(", ")) + (sources.length > limit ? tr(" ({0} in total)", sources.length) : "");
}

function sourceLabel(source: PlotSource): string {
  return `${source.label || source.problem_name || tr("Unnamed")} · ${source.algorithm_name || shortId(source.runset_uid)}`;
}

function optionalFiniteNumber(value: string, label: string): number | undefined {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(tr("{0} must be a finite number.", label));
  return parsed;
}

function axisLabel(axis?: string): string {
  const normalized = canonicalTrendAxis(axis);
  if (["environment", "environment_trend", "env"].includes(normalized)) return tr("Environment index");
  if (["fe", "fes", "fitness_evaluations", "function_evaluations"].includes(normalized)) return tr("Function evaluations (FE)");
  return axis ? axis.replaceAll("_", " ") : tr("Unknown x-axis");
}

function canonicalTrendAxis(axis?: string): string {
  const normalized = (axis ?? "fe").trim().toLowerCase();
  if (["environment", "environment_trend", "env"].includes(normalized)) return "environment";
  if (["fe", "fes", "fitness_evaluations", "function_evaluations"].includes(normalized)) return "fe";
  return normalized;
}

function axisRangeLabel(axis?: string): string {
  return axisLabel(axis).replace(tr("Function evaluations (FE)"), "FE ");
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function errorCode(error: unknown): string | undefined {
  if (!error || typeof error !== "object" || !("code" in error)) return undefined;
  const code = (error as { code?: unknown }).code;
  return typeof code === "string" ? code : undefined;
}

function shortId(value: string): string {
  return value.length > 12 ? `${value.slice(0, 8)}…` : value;
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1024 ** 2).toFixed(1)} MiB`;
}
