import type {
  CreateRunInput,
  DashboardData,
  HealthData,
  ListResponse,
  RemoteRun,
  RemoteServer,
  RemoteSettings,
  ResourceSample,
  ResultCandidate,
  ResultCatalogItem,
  ResultArchiveUploadResponse,
  ResultBundleImportResponse,
  RunResultsPreview,
  RunsetSamplesPreview,
  RunsetTrendPreview,
  SAAnalysis,
  SAAnalysisInput,
  SAImport,
  SAImportResult,
  SAQuickFilterResult,
  SAMatchResult,
  SAMethod,
  RunDetail,
  RunEvent,
  ServerInput,
  ServerResourceHistoryResponse,
  ServerResourceLatestResponse,
  ServerResourceSample,
  TaskFile,
  TaskDocument,
  TaskDeleteResult,
  TaskInspection,
} from "../types";
import type {
  BenchmarkImageResult,
  BenchmarkImmediateResult,
  BenchmarkLimits,
  BenchmarkNbnResult,
  BenchmarkOperationInput,
  BenchmarkProblemSpec,
  BenchmarkSurfaceResult,
} from "../features/benchmark";
import type {
  PlotBoxplotSeries,
  PlotCompletedResult,
  PlotCreateInput,
  PlotPreviewInput,
  PlotPreviewResult,
  PlotSource,
  PlotTrendSeries,
  PlotTrendSeriesRequest,
} from "../features/plots";
import { tr } from "../lib/i18n";

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

const unwrap = <T,>(payload: unknown): T => {
  if (payload && typeof payload === "object" && "data" in payload) {
    return (payload as { data: T }).data;
  }
  return payload as T;
};

const list = <T,>(payload: T[] | ListResponse<T>): T[] => Array.isArray(payload) ? payload : payload.items;

export const request = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (error) {
    throw new ApiError(
      tr("Cannot reach the local EDOLAB Relay API; check that the launcher is still running."),
      0,
      "network_error",
    );
  }
  const body = response.status === 204 ? undefined : await response.json().catch(() => undefined);
  if (!response.ok) {
    const errorBody = body as { detail?: string; message?: string; code?: string } | undefined;
    throw new ApiError(errorBody?.detail ?? errorBody?.message ?? tr("Request failed"), response.status, errorBody?.code);
  }
  return unwrap<T>(body);
};

const requestPlotPreview = async (input: PlotPreviewInput): Promise<PlotPreviewResult> => {
  let response: Response;
  try {
    response = await fetch("/api/plots/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  } catch {
    throw new ApiError(tr("Cannot reach the local EDOLAB Relay API; check that the launcher is still running."), 0, "network_error");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => undefined) as { detail?: string; message?: string; code?: string } | undefined;
    throw new ApiError(body?.detail ?? body?.message ?? tr("Preview could not be generated"), response.status, body?.code);
  }
  const header = (name: string) => response.headers.get(name) ?? "";
  const decodeHeader = (name: string) => {
    const value = header(name);
    try { return decodeURIComponent(value); }
    catch { return value; }
  };
  return {
    kind: header("X-EDOLAB-Plot-Kind") as PlotPreviewResult["kind"],
    metric_name: decodeHeader("X-EDOLAB-Metric-Name"),
    series_count: Number(header("X-EDOLAB-Series-Count")),
    point_count: Number(header("X-EDOLAB-Point-Count")),
    axis: header("X-EDOLAB-Axis") ? decodeHeader("X-EDOLAB-Axis") : undefined,
    image: await response.blob(),
  };
};

const requestResultExport = async (payload: { result_ids?: string[]; selections?: Array<{ result_id: string; runset_uids: string[] }> }): Promise<{ blob: Blob; filename: string }> => {
  let response: Response;
  try {
    response = await fetch("/api/results/exports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(tr("Cannot reach the local EDOLAB Relay API; check that the launcher is still running."), 0, "network_error");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => undefined) as { detail?: string; message?: string; code?: string } | undefined;
    throw new ApiError(body?.detail ?? body?.message ?? tr("Result export failed"), response.status, body?.code);
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const quoted = disposition.match(/filename="([^"]+)"/i)?.[1];
  let filename = encoded ?? quoted ?? "edolab-results.zip";
  try { filename = decodeURIComponent(filename); } catch { /* Keep server-provided filename. */ }
  return { blob: await response.blob(), filename };
};

const requestResultUpload = async (archive: File): Promise<ResultArchiveUploadResponse> => {
  let response: Response;
  try {
    response = await fetch("/api/results/imports/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/zip",
        "X-EDOLAB-Filename": encodeURIComponent(archive.name),
      },
      body: archive,
    });
  } catch {
    throw new ApiError(tr("Cannot reach the local EDOLAB Relay API; check that the launcher is still running."), 0, "network_error");
  }
  const body = await response.json().catch(() => undefined) as ResultArchiveUploadResponse | { detail?: string; message?: string; code?: string } | undefined;
  if (!response.ok) {
    const errorBody = body as { detail?: string; message?: string; code?: string } | undefined;
    throw new ApiError(errorBody?.detail ?? errorBody?.message ?? tr("Result ZIP upload failed"), response.status, errorBody?.code);
  }
  return body as ResultArchiveUploadResponse;
};

const queryString = (values: Record<string, string | number | boolean | undefined>): string => {
  const parameters = new URLSearchParams();
  for (const [name, value] of Object.entries(values)) {
    if (value !== undefined) parameters.set(name, String(value));
  }
  const encoded = parameters.toString();
  return encoded ? "?" + encoded : "";
};

interface BenchmarkCatalogResponse {
  problems: BenchmarkProblemSpec[];
  limits: BenchmarkLimits;
}

let benchmarkCatalogPromise: Promise<BenchmarkCatalogResponse> | null = null;

const benchmarkCatalog = (): Promise<BenchmarkCatalogResponse> => {
  if (!benchmarkCatalogPromise) {
    benchmarkCatalogPromise = request<BenchmarkCatalogResponse>("/api/benchmarks")
      .catch((error) => {
        benchmarkCatalogPromise = null;
        throw error;
      });
  }
  return benchmarkCatalogPromise;
};

const benchmarkSubmit = <T extends BenchmarkImmediateResult>(
  operation: "contour" | "surface" | "nbn_views" | "nbn_model",
  input: BenchmarkOperationInput,
): Promise<T> => request<T>(
  "/api/benchmarks/" + operation,
  { method: "POST", body: JSON.stringify(input) },
);

export const api = {
  health: () => request<HealthData>("/api/health"),
  dashboard: () => request<DashboardData>("/api/dashboard"),
  servers: {
    list: async () => list(await request<RemoteServer[] | ListResponse<RemoteServer>>("/api/servers")),
    create: (input: ServerInput) => request<RemoteServer>("/api/servers", { method: "POST", body: JSON.stringify(input) }),
    update: (id: string, input: ServerInput) => request<RemoteServer>("/api/servers/" + encodeURIComponent(id), { method: "PUT", body: JSON.stringify(input) }),
    remove: (id: string) => request<void>("/api/servers/" + encodeURIComponent(id), { method: "DELETE" }),
    probe: (id: string) => request<RemoteServer>("/api/servers/" + encodeURIComponent(id) + "/probe", { method: "POST" }),
    latestResources: () => request<ServerResourceLatestResponse>("/api/servers/resources/latest"),
    resourceHistory: (id: string, { since, until, afterId, limit = 180 }: { since?: string; until?: string; afterId?: number; limit?: number } = {}) => request<ServerResourceHistoryResponse>(
      "/api/servers/" + encodeURIComponent(id) + "/resource-samples" + queryString({ since, until, after_id: afterId, limit }),
    ),
    refreshResources: (id: string, signal?: AbortSignal) => request<ServerResourceSample>("/api/servers/" + encodeURIComponent(id) + "/resources/refresh", { method: "POST", headers: { "X-EDOLAB-Manual-Sample": "1" }, signal }),
  },
  tasks: {
    list: async () => list(await request<TaskFile[] | ListResponse<TaskFile>>("/api/tasks")),
    inspect: (input: CreateRunInput) => request<TaskInspection>("/api/tasks/inspect", { method: "POST", body: JSON.stringify(input) }),
    get: (filename: string) => request<TaskDocument>("/api/task-files/" + encodeURIComponent(filename)),
    create: (filename: string, content: string) => request<TaskDocument>("/api/task-files", { method: "POST", body: JSON.stringify({ filename, content }) }),
    update: (filename: string, content: string) => request<TaskDocument>("/api/task-files/" + encodeURIComponent(filename), { method: "PUT", body: JSON.stringify({ content }) }),
    remove: (filename: string) => request<TaskDeleteResult>("/api/task-files/" + encodeURIComponent(filename), { method: "DELETE" }),
  },
  runs: {
    list: async ({ limit = 100, offset = 0, status = "all", search }: { limit?: number; offset?: number; status?: string; search?: string } = {}) => list(
      await request<RemoteRun[] | ListResponse<RemoteRun>>(
        "/api/runs" + queryString({ limit, offset, status, search }),
      ),
    ),
    create: (input: CreateRunInput) => request<RemoteRun>("/api/runs", { method: "POST", body: JSON.stringify(input) }),
    get: (id: string) => request<RunDetail>("/api/runs/" + encodeURIComponent(id)),
    rename: (id: string, displayName: string) => request<RemoteRun>("/api/runs/" + encodeURIComponent(id) + "/display-name", { method: "PUT", body: JSON.stringify({ display_name: displayName }) }),
    events: async (id: string, { afterSequence = 0, limit = 1_500 }: { afterSequence?: number; limit?: number } = {}) => list(
      await request<RunEvent[] | ListResponse<RunEvent>>(
        "/api/runs/" + encodeURIComponent(id) + "/events" + queryString({ after_sequence: afterSequence, limit }),
      ),
    ),
    samples: async (id: string, { afterId, limit = 1_000 }: { afterId?: number; limit?: number } = {}) => list(
      await request<ResourceSample[] | ListResponse<ResourceSample>>(
        "/api/runs/" + encodeURIComponent(id) + "/samples" + queryString({ after_id: afterId, limit }),
      ),
    ),
    cancel: (id: string) => request<RemoteRun>("/api/runs/" + encodeURIComponent(id) + "/cancel", { method: "POST" }),
    recover: (id: string) => request<RunDetail>("/api/runs/" + encodeURIComponent(id) + "/recover", { method: "POST" }),
    remove: (id: string) => request<void>("/api/runs/" + encodeURIComponent(id), { method: "DELETE" }),
  },
  results: {
    candidates: async () => list(await request<ResultCandidate[] | ListResponse<ResultCandidate>>("/api/results/candidates")),
    catalog: async () => list(await request<ResultCatalogItem[] | ListResponse<ResultCatalogItem>>("/api/results/catalog")),
    importBundle: (sourcePath: string, displayName?: string) => request<ResultBundleImportResponse>("/api/results/imports", {
      method: "POST",
      body: JSON.stringify({ source_path: sourcePath, ...(displayName?.trim() ? { display_name: displayName.trim() } : {}) }),
    }),
    uploadArchive: (archive: File) => requestResultUpload(archive),
    importCandidate: (candidateId: string, displayName?: string) => request<ResultArchiveUploadResponse>("/api/results/imports/from-candidate", {
      method: "POST",
      body: JSON.stringify({ candidate_id: candidateId, ...(displayName?.trim() ? { display_name: displayName.trim() } : {}) }),
    }),
    preview: (resultId: string) => request<RunResultsPreview>("/api/results/" + encodeURIComponent(resultId)),
    samples: (resultId: string, runsetUid: string, { offset = 0, limit = 200 }: { offset?: number; limit?: number } = {}) => request<RunsetSamplesPreview>(
      "/api/results/" + encodeURIComponent(resultId) + "/runsets/" + encodeURIComponent(runsetUid) + "/samples" + queryString({ offset, limit }),
    ),
    trend: (resultId: string, runsetUid: string, metricName: string, { offset = 0, limit = 1_000 }: { offset?: number; limit?: number } = {}) => request<RunsetTrendPreview>(
      "/api/results/" + encodeURIComponent(resultId) + "/runsets/" + encodeURIComponent(runsetUid) + "/trends/" + encodeURIComponent(metricName) + queryString({ offset, limit }),
    ),
    remove: (resultId: string) => request<void>("/api/results/" + encodeURIComponent(resultId), { method: "DELETE" }),
    exportArchive: (resultIds: string[]) => requestResultExport({ result_ids: resultIds }),
    exportRunsets: (selections: Array<{ result_id: string; runset_uids: string[] }>) => requestResultExport({ selections }),
  },
  plots: {
    listSources: () => request<PlotSource[]>("/api/plots/sources"),
    loadTrendSeries: (sourceId: string, metricName: string, { xMin, xMax, maxPoints }: PlotTrendSeriesRequest = {}) => request<Omit<PlotTrendSeries, "label">>(
      "/api/plots/series/trend" + queryString({ source_id: sourceId, metric_name: metricName, x_min: xMin, x_max: xMax, max_points: maxPoints }),
    ),
    loadBoxplotSeries: (sourceId: string, metricName: string) => request<Omit<PlotBoxplotSeries, "label">>(
      "/api/plots/series/boxplot" + queryString({ source_id: sourceId, metric_name: metricName }),
    ),
    createPreview: (input: PlotPreviewInput) => requestPlotPreview(input),
    createPlot: (input: PlotCreateInput) => request<PlotCompletedResult>("/api/plots", { method: "POST", body: JSON.stringify(input) }),
    artifactUrl: (artifact: { url?: string }) => artifact.url ?? "",
  },
  benchmarks: {
    listProblems: async () => (await benchmarkCatalog()).problems,
    limits: async () => (await benchmarkCatalog()).limits,
    renderContour: (input: BenchmarkOperationInput) => benchmarkSubmit<BenchmarkImageResult>("contour", input),
    buildSurface: (input: BenchmarkOperationInput) => benchmarkSubmit<BenchmarkSurfaceResult>("surface", input),
    renderNbnViews: (input: BenchmarkOperationInput) => benchmarkSubmit<BenchmarkImageResult>("nbn_views", input),
    buildNbnModel: (input: BenchmarkOperationInput) => benchmarkSubmit<BenchmarkNbnResult>("nbn_model", input),
    artifactUrl: (artifact: { url?: string }) => artifact.url ?? "",
  },
  sa: {
    imports: async () => list(await request<SAImport[] | ListResponse<SAImport>>("/api/sa/imports")),
    importRunsets: (resultId: string, runsetUids?: string[], allowDuplicates = false) => request<SAImportResult>("/api/sa/imports", { method: "POST", body: JSON.stringify({ result_id: resultId, run_id: resultId, runset_uids: runsetUids, allow_duplicates: allowDuplicates }) }),
    importSelections: (selections: Array<{ result_id: string; runset_uids: string[] }>, allowDuplicates = false) => request<SAImportResult>("/api/sa/imports", { method: "POST", body: JSON.stringify({ selections, allow_duplicates: allowDuplicates }) }),
    quickFilter: (allowDuplicates = false) => request<SAQuickFilterResult>("/api/sa/quick-filter", { method: "POST", body: JSON.stringify({ allow_duplicates: allowDuplicates }) }),
    removeImport: (id: string) => request<void>("/api/sa/imports/" + encodeURIComponent(id), { method: "DELETE" }),
    clearImports: () => request<void>("/api/sa/imports", { method: "DELETE" }),
    match: (method: SAMethod, baseImportId?: string) => request<SAMatchResult>("/api/sa/match", { method: "POST", body: JSON.stringify({ method, base_import_id: baseImportId }) }),
    analyses: async () => list(await request<SAAnalysis[] | ListResponse<SAAnalysis>>("/api/sa/analyses")),
    createAnalysis: (input: SAAnalysisInput) => request<SAAnalysis>("/api/sa/analyses", { method: "POST", body: JSON.stringify(input) }),
    getAnalysis: (id: string) => request<SAAnalysis>("/api/sa/analyses/" + encodeURIComponent(id)),
    removeAnalysis: (id: string) => request<void>("/api/sa/analyses/" + encodeURIComponent(id), { method: "DELETE" }),
  },
  settings: {
    get: () => request<RemoteSettings>("/api/settings"),
    save: (input: RemoteSettings) => request<RemoteSettings>("/api/settings", { method: "PUT", body: JSON.stringify(input) }),
  },
};
