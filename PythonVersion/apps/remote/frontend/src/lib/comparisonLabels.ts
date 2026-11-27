import { tr } from "./i18n";
export type ComparisonLabelMode = "algorithm" | `algorithmParam:${string}`;

export interface ComparisonLabelSource {
  id?: string;
  import_id?: string;
  algorithm_name: string;
  algorithm_params?: Record<string, unknown>;
  runset_uid: string;
}

export interface ComparisonLabelPlan {
  mode: ComparisonLabelMode;
  kind: "algorithm" | "algorithm_parameter";
  title: string;
  parameter?: string;
  parameterModes: Array<`algorithmParam:${string}`>;
  labels: string[];
  mixedAlgorithms: boolean;
}

export function comparisonLabelPlan(
  sources: ComparisonLabelSource[],
  preferredMode: ComparisonLabelMode = "algorithm",
): ComparisonLabelPlan {
  const algorithms = new Set(sources.map((source) => source.algorithm_name));
  const mixedAlgorithms = algorithms.size > 1;
  const parameterModes = varyingAlgorithmParameterKeys(sources).map(
    (key) => `algorithmParam:${key}` as const,
  );
  const mode = mixedAlgorithms || !parameterModes.length
    ? "algorithm"
    : parameterModes.includes(preferredMode as `algorithmParam:${string}`)
      ? preferredMode
      : parameterModes[0];
  const parameter = mode.startsWith("algorithmParam:")
    ? mode.slice("algorithmParam:".length)
    : undefined;
  const baseLabels = sources.map((source) => parameter
    ? formatComparisonValue(source.algorithm_params?.[parameter])
    : source.algorithm_name || tr("Unnamed algorithm"));

  return {
    mode,
    kind: parameter ? "algorithm_parameter" : "algorithm",
    title: parameter ?? tr("Algorithm"),
    parameter,
    parameterModes,
    labels: uniqueComparisonLabels(baseLabels, sources),
    mixedAlgorithms,
  };
}

export function varyingAlgorithmParameterKeys(sources: ComparisonLabelSource[]): string[] {
  if (sources.length < 2 || new Set(sources.map((source) => source.algorithm_name)).size !== 1) return [];
  const common = new Set(Object.keys(sources[0].algorithm_params ?? {}));
  for (const source of sources.slice(1)) {
    const keys = new Set(Object.keys(source.algorithm_params ?? {}));
    for (const key of common) if (!keys.has(key)) common.delete(key);
  }
  return [...common]
    .filter((key) => new Set(sources.map((source) => stableComparisonValue(source.algorithm_params?.[key]))).size > 1)
    .sort((left, right) => left.localeCompare(right));
}

export function formatComparisonValue(value: unknown): string {
  if (value === undefined) return "—";
  if (value === null) return "null";
  if (typeof value === "string") return value || '""';
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try { return JSON.stringify(canonicalValue(value)); }
  catch { return String(value); }
}

function uniqueComparisonLabels(baseLabels: string[], sources: ComparisonLabelSource[]): string[] {
  const baseCounts = countLabels(baseLabels);
  const runsetLabels = baseLabels.map((label, index) => (
    (baseCounts.get(label) ?? 0) > 1
      ? `${label} · ${shortRunsetId(sources[index].runset_uid)}`
      : label
  ));
  const runsetCounts = countLabels(runsetLabels);
  const sourceLabels = runsetLabels.map((label, index) => (
    (runsetCounts.get(label) ?? 0) > 1
      ? `${label} · ${shortSourceId(sourceIdentity(sources[index], index))}`
      : label
  ));
  return ensureUniqueLabels(sourceLabels);
}

function countLabels(labels: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  return counts;
}

function sourceIdentity(source: ComparisonLabelSource, index: number): string {
  return source.import_id?.trim()
    || source.id?.trim()
    || source.runset_uid.trim()
    || `source-${index + 1}`;
}

function shortSourceId(value: string): string {
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-4)}` : value;
}

function ensureUniqueLabels(labels: string[]): string[] {
  const used = new Set<string>();
  return labels.map((label) => {
    if (!used.has(label)) {
      used.add(label);
      return label;
    }
    let suffix = 2;
    let candidate = `${label} · #${suffix}`;
    while (used.has(candidate)) {
      suffix += 1;
      candidate = `${label} · #${suffix}`;
    }
    used.add(candidate);
    return candidate;
  });
}

function stableComparisonValue(value: unknown): string {
  try { return JSON.stringify(canonicalValue(value)); }
  catch { return String(value); }
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalValue(item)]),
    );
  }
  return value;
}

function shortRunsetId(value: string): string {
  return value.length > 10 ? value.slice(0, 10) : value;
}
