import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Image as ImageIcon,
  Network,
  Play,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { Badge, Button, EmptyState, LoadingBlock, Notice, PageHeader, Tabs } from "../../components/ui";
import { InteractiveNbn } from "./InteractiveNbn";
import { InteractiveSurface } from "./InteractiveSurface";
import "./benchmark.css";
import { tr, LOCALE } from "../../lib/i18n";

export interface BenchmarkParameterSpec {
  name: string;
  default: unknown;
  kind: "integer" | "numeric" | "boolean" | "option" | "string" | string;
  bounds?: unknown;
  description?: string;
}

export interface BenchmarkProblemSpec {
  name: string;
  parameters: BenchmarkParameterSpec[];
  default_dimension?: number | null;
  supports_2d?: boolean;
  supports_evaluation_device?: boolean;
}

export interface BenchmarkLimits {
  contour_resolution_min: number;
  contour_resolution_max: number;
  surface_resolution_min: number;
  surface_resolution_max: number;
  nbn_sample_count_min: number;
  nbn_sample_count_max: number;
}

export interface BenchmarkOperationInput {
  problem_name: string;
  problem_params: Record<string, unknown>;
  environment_number: number;
  seed: number | null;
  resolution: number;
  sample_count: number;
}

export interface BenchmarkArtifact {
  id: string;
  storage_key: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  url?: string;
}

interface BenchmarkResultBase {
  kind: string;
  problem_name: string;
  dimension: number;
  environment: number;
  seed?: number | null;
  problem_params: Record<string, unknown>;
}

export interface BenchmarkImageResult extends BenchmarkResultBase {
  kind: "contour" | "nbn_three_views";
  artifact: BenchmarkArtifact;
}

export interface BenchmarkSurfaceResult extends BenchmarkResultBase {
  kind: "surface";
  x: number[];
  y: number[];
  z: Array<Array<number | null>>;
  z_min: number;
  z_max: number;
  slice_center: number[];
}

export interface BenchmarkNbnResult extends BenchmarkResultBase {
  kind: "nbn_model";
  nodes: Array<{
    id: number;
    x: number;
    y: number;
    z: number;
    fitness: number;
    normalized_fitness: number;
    basin: number;
    depth: number;
    is_root: boolean;
    sample: number[];
  }>;
  edges: Array<{ source: number; target: number }>;
  summary: Record<string, unknown>;
}

export type BenchmarkImmediateResult = BenchmarkImageResult | BenchmarkSurfaceResult | BenchmarkNbnResult;

export interface BenchmarkApi {
  listProblems: () => Promise<BenchmarkProblemSpec[]>;
  limits?: () => Promise<BenchmarkLimits>;
  renderContour: (input: BenchmarkOperationInput) => Promise<BenchmarkImageResult>;
  buildSurface: (input: BenchmarkOperationInput) => Promise<BenchmarkSurfaceResult>;
  renderNbnViews: (input: BenchmarkOperationInput) => Promise<BenchmarkImageResult>;
  buildNbnModel: (input: BenchmarkOperationInput) => Promise<BenchmarkNbnResult>;
  artifactUrl?: (artifact: BenchmarkArtifact) => string;
}

export interface BenchmarkPageProps {
  api: BenchmarkApi;
  refreshNonce?: number;
}

type Operation = "contour" | "surface" | "nbn_views" | "nbn_model";

const defaultLimits: BenchmarkLimits = {
  contour_resolution_min: 8,
  contour_resolution_max: 240,
  surface_resolution_min: 8,
  surface_resolution_max: 140,
  nbn_sample_count_min: 20,
  nbn_sample_count_max: 100_000,
};

const operationCopy: Record<Operation, { label: string; busy: string }> = {
  contour: { label: tr("2D contour"), busy: tr("Drawing contours…") },
  surface: { label: tr("Interactive surface"), busy: tr("Building the surface…") },
  nbn_views: { label: tr("NBN three views"), busy: tr("Drawing the NBN views…") },
  nbn_model: { label: tr("Interactive NBN"), busy: tr("Building the NBN…") },
};

const operationOrder: Operation[] = ["contour", "surface", "nbn_views", "nbn_model"];

export function BenchmarkPage({
  api,
  refreshNonce = 0,
}: BenchmarkPageProps) {
  const [problems, setProblems] = useState<BenchmarkProblemSpec[]>([]);
  const [limits, setLimits] = useState(defaultLimits);
  const [problemName, setProblemName] = useState("");
  const [paramsText, setParamsText] = useState("{}");
  const [environment, setEnvironment] = useState("1");
  const [seed, setSeed] = useState("1");
  const [resolution, setResolution] = useState("80");
  const [sampleCount, setSampleCount] = useState("800");
  const [results, setResults] = useState<Partial<Record<Operation, BenchmarkImmediateResult>>>({});
  const [activePreview, setActivePreview] = useState<Operation | null>(null);
  const [busy, setBusy] = useState<Operation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const previewScope = useRef(0);

  useEffect(() => {
    let active = true;
    setLoading(true);
    Promise.all([api.listProblems(), api.limits?.() ?? Promise.resolve(defaultLimits)])
      .then(([nextProblems, nextLimits]) => {
        if (!active) return;
        setProblems(nextProblems);
        setLimits(nextLimits);
        const nextName = nextProblems.some((item) => item.name === problemName)
          ? problemName
          : nextProblems[0]?.name ?? "";
        setProblemName(nextName);
        if (nextName !== problemName) {
          const selected = nextProblems.find((item) => item.name === nextName);
          if (selected) setParamsText(JSON.stringify(defaultParameters(selected, true), null, 2));
          previewScope.current += 1;
          setResults({});
          setActivePreview(null);
        }
        setError(null);
      })
      .catch((nextError: unknown) => {
        if (active) setError(errorMessage(nextError, tr("Could not read the problem list")));
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [api, refreshNonce]);

  const selectedProblem = problems.find((item) => item.name === problemName);
  const parsedParams = useMemo(() => parseObject(paramsText), [paramsText]);
  const dimension = parsedParams.value ? Number(parsedParams.value.Dimension) : Number.NaN;
  const isTwoDimensional = Number.isFinite(dimension) && dimension === 2;
  const environmentMax = parsedParams.value ? Number(parsedParams.value.EnvironmentNumber ?? 1) : 1;
  const availablePreviews = operationOrder.filter((operation) => Boolean(results[operation]));
  const visibleOperation = activePreview && results[activePreview] ? activePreview : availablePreviews[0] ?? null;
  const visibleResult = visibleOperation ? results[visibleOperation] ?? null : null;

  const selectProblem = (name: string, prefer2d = true) => {
    const problem = problems.find((item) => item.name === name);
    const changed = name !== problemName;
    setProblemName(name);
    if (problem) setParamsText(JSON.stringify(defaultParameters(problem, prefer2d), null, 2));
    if (changed) {
      previewScope.current += 1;
      setResults({});
      setActivePreview(null);
    }
    setError(null);
  };

  const run = async (operation: Operation) => {
    setError(null);
    if (!selectedProblem) {
      setError(tr("Choose a test problem first."));
      return;
    }
    if (parsedParams.error || !parsedParams.value) {
      setError(parsedParams.error ?? tr("Problem parameters must be a JSON object."));
      return;
    }
    if ((operation === "contour" || operation === "surface") && !isTwoDimensional) {
      setError(tr("2D contours and surfaces need Dimension = 2; use NBN for higher dimensions."));
      return;
    }
    let input: BenchmarkOperationInput;
    try {
      const needsResolution = operation === "contour" || operation === "surface";
      const needsSampleCount = operation === "nbn_views" || operation === "nbn_model";
      input = {
        problem_name: problemName,
        problem_params: parsedParams.value,
        environment_number: positiveInteger(environment, tr("Environment index")),
        seed: optionalNonNegativeInteger(seed, tr("Random seed")),
        resolution: needsResolution
          ? positiveInteger(resolution, tr("Resolution"))
          : limits.surface_resolution_min,
        sample_count: needsSampleCount
          ? positiveInteger(sampleCount, tr("NBN samples"))
          : limits.nbn_sample_count_min,
      };
    } catch (nextError) {
      setError(errorMessage(nextError, tr("Invalid parameters")));
      return;
    }
    const requestScope = previewScope.current;
    setBusy(operation);
    try {
      const response = operation === "contour"
        ? await api.renderContour(input)
        : operation === "surface"
          ? await api.buildSurface(input)
          : operation === "nbn_views"
            ? await api.renderNbnViews(input)
            : await api.buildNbnModel(input);
      if (requestScope !== previewScope.current) return;
      setResults((current) => ({ ...current, [operation]: response }));
      setActivePreview(operation);
    } catch (nextError) {
      setError(errorMessage(nextError, tr("Problem plot failed")));
    } finally {
      setBusy(null);
    }
  };

  if (loading) return <LoadingBlock label={tr("Reading the problems that can be drawn…")} />;

  return (
    <div className="page benchmark-workspace">
      <PageHeader
        title={tr("Problems and NBN")}
        description={tr("2D contours, interactive surfaces and NBN topology computed on this computer")}
      />

      {error ? <Notice tone="danger"><strong>{tr("Cannot run")}</strong><p>{error}</p></Notice> : null}

      <div className="benchmark-layout">
        <section className="panel benchmark-controls" aria-label={tr("Problem plot settings")}>
          <header className="panel__header">
            <h2>{tr("Problem and plot parameters")}</h2>
            <Button
              tone="quiet"
              icon={<RotateCcw size={14} />}
              disabled={!selectedProblem || Boolean(busy)}
              onClick={() => selectedProblem && selectProblem(selectedProblem.name, true)}
            >{tr("Restore defaults")}</Button>
          </header>

          <div className="benchmark-controls__body">
            <div className="benchmark-fields">
              <label>
                <span>{tr("Test problem")}</span>
                <select value={problemName} onChange={(event) => selectProblem(event.target.value)}>
                  {problems.map((problem) => <option key={problem.name} value={problem.name}>{problem.name}</option>)}
                </select>
                <small>{isTwoDimensional ? tr("2D landscape can be drawn") : tr("Dimension {0}; NBN can be computed", Number.isFinite(dimension) ? dimension : tr("Unknown"))}</small>
              </label>
              <label>
                <span>{tr("Environment index")}</span>
                <input type="number" min={1} max={Number.isFinite(environmentMax) ? environmentMax : undefined} value={environment} onChange={(event) => setEnvironment(event.target.value)} />
                <small>{tr("Range 1–{0}", Number.isFinite(environmentMax) ? environmentMax : "?")}</small>
              </label>
              <label>
                <span>{tr("Random seed")}</span>
                <input type="number" min={0} value={seed} onChange={(event) => setSeed(event.target.value)} placeholder={tr("Random if left empty")} />
                <small>{tr("A fixed seed reproduces the same problem environment")}</small>
              </label>
              <label>
                <span>{tr("Plot resolution")}</span>
                <input type="number" min={limits.contour_resolution_min} max={limits.contour_resolution_max} value={resolution} onChange={(event) => setResolution(event.target.value)} />
                <small>{tr("At most {0} for surfaces; at most {1} for contours", limits.surface_resolution_max, limits.contour_resolution_max)}</small>
              </label>
              <label>
                <span>{tr("NBN samples")}</span>
                <input type="number" min={limits.nbn_sample_count_min} max={limits.nbn_sample_count_max} value={sampleCount} onChange={(event) => setSampleCount(event.target.value)} />
                <small>{tr("At most {0}; larger numbers take longer", limits.nbn_sample_count_max.toLocaleString(LOCALE))}</small>
              </label>
            </div>

            <label className="benchmark-json-field">
              <span>{tr("Problem parameters (JSON)")}</span>
              <textarea spellCheck={false} value={paramsText} onChange={(event) => setParamsText(event.target.value)} />
              <small className={parsedParams.error ? "is-error" : ""}>{parsedParams.error ?? tr("The parameters are used only for this local plot; no code snapshot or run record is created.")}</small>
            </label>

            <div className="benchmark-actions" aria-label={tr("Plot type")}>
              <Button tone="primary" icon={<ImageIcon size={15} />} disabled={Boolean(busy) || !isTwoDimensional} onClick={() => void run("contour")}>
                {busy === "contour" ? <><RefreshCw className="is-spinning" size={14} />{tr("Drawing…")}</> : operationCopy.contour.label}
              </Button>
              <Button icon={<Box size={15} />} disabled={Boolean(busy) || !isTwoDimensional} onClick={() => void run("surface")}>
                {busy === "surface" ? tr("Building…") : operationCopy.surface.label}
              </Button>
              <Button icon={<Network size={15} />} disabled={Boolean(busy)} onClick={() => void run("nbn_views")}>
                {busy === "nbn_views" ? tr("Drawing…") : operationCopy.nbn_views.label}
              </Button>
              <Button icon={<Play size={15} />} disabled={Boolean(busy)} onClick={() => void run("nbn_model")}>
                {busy === "nbn_model" ? tr("Building…") : operationCopy.nbn_model.label}
              </Button>
            </div>
            {busy ? <p className="benchmark-busy"><RefreshCw className="is-spinning" size={14} />{operationCopy[busy].busy}{" "}{tr("The result is added to the preview on the right when it is done.")}</p> : null}
          </div>
        </section>

        <section className="panel benchmark-preview" aria-label={tr("Plot preview")}>
          <header className="panel__header">
            <h2>{tr("Structure preview")}</h2>
            {visibleResult ? <Badge tone="info" className="benchmark-result-pill">{tr("{0} · Env {1} · {2} views", visibleResult.problem_name, visibleResult.environment, availablePreviews.length)}</Badge> : null}
          </header>
          {availablePreviews.length && visibleOperation ? (
            <Tabs
              variant="underline"
              className="benchmark-preview-tabs"
              ariaLabel={tr("Generated structure views")}
              value={visibleOperation}
              onChange={setActivePreview}
              items={availablePreviews.map((operation) => ({
                value: operation,
                id: `benchmark-tab-${operation}`,
                controls: `benchmark-panel-${operation}`,
                label: operationCopy[operation].label,
              }))}
            />
          ) : null}
          <div className="benchmark-preview__stage">
            {availablePreviews.length ? availablePreviews.map((operation) => {
              const previewResult = results[operation];
              if (!previewResult) return null;
              return (
                <div
                  key={operation}
                  id={`benchmark-panel-${operation}`}
                  role="tabpanel"
                  aria-labelledby={`benchmark-tab-${operation}`}
                  className="benchmark-preview-panel"
                  hidden={visibleOperation !== operation}
                >
                  <BenchmarkPreview
                    result={previewResult}
                    api={api}
                    active={visibleOperation === operation}
                  />
                </div>
              );
            }) : <BenchmarkPreview result={null} api={api} active />}
          </div>
        </section>
      </div>

      <section className="panel benchmark-specs">
        <header className="panel__header"><h2>{tr("Parameters of {0}", selectedProblem?.name ?? tr("Problem"))}</h2></header>
        {selectedProblem ? <ParameterTable problem={selectedProblem} /> : <EmptyState title={tr("No problems available")} description={tr("Check that the problem registry is loaded.")} />}
      </section>
    </div>
  );
}

function BenchmarkPreview({
  result,
  api,
  active,
}: {
  result: BenchmarkImmediateResult | null;
  api: BenchmarkApi;
  active: boolean;
}) {
  if (!result) {
    return <EmptyState title={tr("No preview yet")} description={tr("Choose a problem and a plot type to generate it on this computer. 2D landscapes are good for visual checks; NBN shows the basin structure in higher dimensions.")} />;
  }
  if (result.kind === "contour" || result.kind === "nbn_three_views") {
    const source = api.artifactUrl?.(result.artifact) ?? result.artifact.url;
    return source
      ? <a href={source} target="_blank" rel="noreferrer" className="benchmark-image-link"><img src={source} alt={`${result.problem_name} ${result.kind}`} /></a>
      : <Notice tone="warning"><p>{tr("The image was generated, but the API returned no accessible artifact URL.")}</p></Notice>;
  }
  if (result.kind === "surface") return <InteractiveSurface result={result} active={active} />;
  if (result.kind === "nbn_model") return <InteractiveNbn result={result} active={active} />;
  return <EmptyState title={tr("Cannot preview the result")} description={tr("Unknown result type: {0}", result.kind)} />;
}

function ParameterTable({ problem }: { problem: BenchmarkProblemSpec }) {
  return (
    <div className="benchmark-spec-table" role="region" aria-label={tr("Parameter table of {0}", problem.name)} tabIndex={0}>
      <table>
        <thead><tr><th>{tr("Parameter")}</th><th>{tr("Default")}</th><th>{tr("Type")}</th><th>{tr("Range / options")}</th><th>{tr("Description")}</th></tr></thead>
        <tbody>
          {problem.parameters.map((parameter) => (
            <tr key={parameter.name}>
              <td><code>{parameter.name}</code></td>
              <td>{formatValue(parameter.default)}</td>
              <td>{parameter.kind}</td>
              <td>{formatValue(parameter.bounds)}</td>
              <td>{parameter.description || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function defaultParameters(problem: BenchmarkProblemSpec, prefer2d: boolean): Record<string, unknown> {
  return Object.fromEntries(problem.parameters.map((parameter) => {
    const value = prefer2d && parameter.name === "Dimension" && boundsInclude(parameter.bounds, 2)
      ? 2
      : parameter.default;
    return [parameter.name, value];
  }));
}

function boundsInclude(bounds: unknown, value: number): boolean {
  return Array.isArray(bounds) && bounds.length === 2 && Number(bounds[0]) <= value && value <= Number(bounds[1]);
}

function parseObject(text: string): { value: Record<string, unknown> | null; error: string | null } {
  try {
    const value: unknown = JSON.parse(text || "{}");
    if (!value || typeof value !== "object" || Array.isArray(value)) return { value: null, error: tr("Problem parameters must be a JSON object.") };
    return { value: value as Record<string, unknown>, error: null };
  } catch (error) {
    return { value: null, error: errorMessage(error, tr("Invalid JSON")) };
  }
}

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new Error(tr("{0} must be a positive integer.", label));
  return parsed;
}

function optionalNonNegativeInteger(value: string, label: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error(tr("{0} must be a non-negative integer.", label));
  return parsed;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "string") return value;
  try { return JSON.stringify(value); } catch { return String(value); }
}
