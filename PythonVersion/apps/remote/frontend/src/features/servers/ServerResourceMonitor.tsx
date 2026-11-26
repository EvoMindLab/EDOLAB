import { AlertTriangle, CircuitBoard, Clock3, Cpu, Gauge, MemoryStick, PauseCircle, TriangleAlert } from "lucide-react";
import { Badge, Meter, type BadgeTone } from "../../components/ui";
import { formatDateTime, formatRelativeTime } from "../../lib/format";
import type { ServerResourceMonitor as ApiResourceMonitor, ServerResourceSample as ApiResourceSample } from "../../types";
import { tr } from "../../lib/i18n";

export type ResourceMonitorState = "fresh" | "stale" | "failed" | "never" | "paused";

export interface ResourceMonitorGpu {
  index: number;
  name: string;
  utilizationPercent: number | null;
  memoryUsedMb: number | null;
  memoryTotalMb: number | null;
  temperatureC?: number | null;
}

export interface ResourceMonitorSample {
  id: number | string;
  sampledAt: string;
  reachable: boolean;
  cpuPercent: number | null;
  memoryPercent: number | null;
  load1m: number | null;
  loadRatio: number | null;
  gpuUsable: boolean | null;
  gpuError?: string | null;
  gpus: ResourceMonitorGpu[];
  error?: string | null;
}

export interface ResourceMonitorViewModel {
  state: ResourceMonitorState;
  sampleIntervalSeconds: number;
  staleAfterSeconds: number;
  latestAttemptAt?: string | null;
  lastSuccessAt?: string | null;
  lastError?: string | null;
  pauseReason?: string | null;
  latestSuccess?: ResourceMonitorSample | null;
  samples: ResourceMonitorSample[];
}

const sampleFromApi = (sample: ApiResourceSample): ResourceMonitorSample => ({
  id: sample.id,
  sampledAt: sample.sampled_at,
  reachable: sample.reachable,
  cpuPercent: sample.cpu_percent,
  memoryPercent: sample.memory_percent,
  load1m: sample.load_1m,
  loadRatio: sample.load_ratio,
  gpuUsable: sample.gpu_usable ?? null,
  gpuError: sample.gpu_error,
  error: sample.error,
  gpus: sample.gpus.map((gpu) => ({
    index: gpu.index,
    name: gpu.name,
    utilizationPercent: gpu.utilization_percent,
    memoryUsedMb: gpu.memory_used_mb,
    memoryTotalMb: gpu.memory_total_mb,
    temperatureC: gpu.temperature_celsius,
  })),
});

export const resourceMonitorFromApi = (
  monitor?: ApiResourceMonitor | null,
  samples: ApiResourceSample[] = [],
): ResourceMonitorViewModel => ({
  state: monitor?.state ?? "never",
  sampleIntervalSeconds: monitor?.sample_interval_seconds ?? 3,
  staleAfterSeconds: monitor?.stale_after_seconds ?? 15,
  latestAttemptAt: monitor?.latest_attempt?.sampled_at,
  lastSuccessAt: monitor?.latest_success?.sampled_at,
  lastError: monitor?.latest_attempt?.reachable === false ? monitor.latest_attempt.error : null,
  pauseReason: monitor?.pause_reason,
  latestSuccess: monitor?.latest_success ? sampleFromApi(monitor.latest_success) : null,
  samples: samples.map(sampleFromApi),
});

const stateCopy: Record<ResourceMonitorState, { label: string; detail: string }> = {
  fresh: { label: tr("Sampling normally"), detail: tr("The background sampler keeps writing to the local database.") },
  stale: { label: tr("Stale data"), detail: tr("The last successful values are still shown, but they should not be taken as the current load.") },
  failed: { label: tr("Last sampling failed"), detail: tr("The last successful values are kept until the background sampler recovers.") },
  never: { label: tr("No samples yet"), detail: tr("Resource data appears here after the environment check and once background monitoring is enabled.") },
  paused: { label: tr("Sampling paused"), detail: tr("The background sampler is not running or was paused by the server; this does not affect whether the machine can be scheduled.") },
};

const monitorStateDetail = (monitor: ResourceMonitorViewModel): string => {
  if (monitor.state !== "paused") return stateCopy[monitor.state].detail;
  if (monitor.pauseReason === "pinned_fingerprint_required") {
    return tr("The full probe has not pinned the machine identity yet; background sampling starts after the identity is verified.");
  }
  if (monitor.pauseReason === "successful_probe_required") {
    return tr("Background resource sampling starts after one successful environment check.");
  }
  return stateCopy.paused.detail;
};

const finite = (value: number | null | undefined): value is number => Number.isFinite(value);
const clamp = (value: number, lower: number, upper: number) => Math.max(lower, Math.min(upper, value));

const percentageLabel = (value: number | null | undefined): string => finite(value) ? `${Math.round(value)}%` : "—";

const memoryLabel = (usedMb: number | null, totalMb: number | null): string => {
  if (!finite(usedMb) || !finite(totalMb) || totalMb <= 0) return tr("GPU memory not reported");
  return `${(usedMb / 1024).toFixed(1)} / ${(totalMb / 1024).toFixed(1)} GB`;
};

const freshnessTone: Record<ResourceMonitorState, BadgeTone> = {
  fresh: "success",
  stale: "warning",
  failed: "danger",
  never: "neutral",
  paused: "neutral",
};

export function ResourceFreshnessBadge({ state }: { state: ResourceMonitorState }) {
  return <Badge tone={freshnessTone[state]} dot className="resource-freshness">{stateCopy[state].label}</Badge>;
}

export function ResourceMiniSummary({ monitor }: { monitor?: ResourceMonitorViewModel | null }) {
  const latest = monitor?.latestSuccess;
  const gpuValues = latest?.gpus.map((gpu) => gpu.utilizationPercent).filter(finite) ?? [];
  const gpuAverage = gpuValues.length ? gpuValues.reduce((sum, value) => sum + value, 0) / gpuValues.length : null;
  return (
    <span className={`resource-mini-summary resource-mini-summary--${monitor?.state ?? "never"}`}>
      <span><Cpu size={14} />CPU <strong>{percentageLabel(latest?.cpuPercent)}</strong></span>
      <span><MemoryStick size={14} />{tr("Memory")}{" "}<strong>{percentageLabel(latest?.memoryPercent)}</strong></span>
      <span><Gauge size={14} />load <strong>{finite(latest?.load1m) ? latest.load1m.toFixed(2) : "—"}</strong></span>
      <span><CircuitBoard size={14} />GPU <strong>{percentageLabel(gpuAverage)}</strong></span>
    </span>
  );
}

function MonitorNotice({ monitor }: { monitor: ResourceMonitorViewModel }) {
  if (monitor.state === "fresh") return null;
  const icon = monitor.state === "failed" ? <TriangleAlert size={16} /> : monitor.state === "paused" ? <PauseCircle size={16} /> : <AlertTriangle size={16} />;
  const error = monitor.lastError?.trim();
  return (
    <div className={`resource-monitor-notice resource-monitor-notice--${monitor.state}`}>
      {icon}
      <div>
        <strong>{stateCopy[monitor.state].label}</strong>
        <p>{error || monitorStateDetail(monitor)}</p>
      </div>
    </div>
  );
}

type MetricKey = "cpuPercent" | "memoryPercent" | "loadRatio";

const metricValue = (sample: ResourceMonitorSample, key: MetricKey): number | null => {
  const raw = sample[key];
  if (!sample.reachable || !finite(raw)) return null;
  return key === "loadRatio" ? raw * 100 : raw;
};

const pointSegments = (
  samples: ResourceMonitorSample[],
  key: MetricKey,
  width: number,
  height: number,
  ceiling: number,
  firstTimestamp: number,
  lastTimestamp: number,
): string[] => {
  const timeRange = Math.max(lastTimestamp - firstTimestamp, 1);
  const indexDenominator = Math.max(samples.length - 1, 1);
  const segments: string[] = [];
  let current: string[] = [];
  samples.forEach((sample, index) => {
    const value = metricValue(sample, key);
    if (value == null) {
      if (current.length > 1) segments.push(current.join(" "));
      current = [];
      return;
    }
    const timestamp = new Date(sample.sampledAt).getTime();
    const x = Number.isFinite(timestamp) && lastTimestamp > firstTimestamp
      ? ((timestamp - firstTimestamp) / timeRange) * width
      : (index / indexDenominator) * width;
    const y = height - (clamp(value, 0, ceiling) / ceiling) * height;
    current.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  if (current.length > 1) segments.push(current.join(" "));
  return segments;
};

function ResourceHistoryChart({ samples }: { samples: ResourceMonitorSample[] }) {
  const ordered = [...samples].sort((left, right) => new Date(left.sampledAt).getTime() - new Date(right.sampledAt).getTime());
  const visible = ordered.slice(-10_000);
  const numericValues = visible.flatMap((sample) => ([sample.cpuPercent, sample.memoryPercent, finite(sample.loadRatio) ? sample.loadRatio * 100 : null])).filter(finite);
  const ceiling = Math.max(100, Math.ceil((Math.max(0, ...numericValues) || 100) / 25) * 25);
  const width = 640;
  const height = 150;
  const firstTimestamp = new Date(visible[0]?.sampledAt ?? "").getTime();
  const lastTimestamp = new Date(visible.at(-1)?.sampledAt ?? "").getTime();
  const series: Array<{ key: MetricKey; className: string; label: string }> = [
    { key: "cpuPercent", className: "is-cpu", label: "CPU" },
    { key: "memoryPercent", className: "is-memory", label: tr("Memory") },
    { key: "loadRatio", className: "is-load", label: tr("load / cores") },
  ];
  const hasDrawableSeries = series.some((item) => visible.filter((sample) => metricValue(sample, item.key) != null).length >= 2);
  const spanMilliseconds = visible.length > 1 && Number.isFinite(firstTimestamp) && Number.isFinite(lastTimestamp) ? Math.max(0, lastTimestamp - firstTimestamp) : 0;
  const spanMinutes = spanMilliseconds / 60_000;
  const spanLabel = spanMinutes >= 60 ? tr("{0} h", (spanMinutes / 60).toFixed(1)) : spanMinutes >= 1 ? tr("{0} min", Math.round(spanMinutes)) : tr("under 1 min");

  if (!hasDrawableSeries) {
    return <div className="resource-history-empty"><Clock3 size={17} /><strong>{tr("Not enough history")}</strong><p>{tr("The load curve appears after at least two successful samples.")}</p></div>;
  }

  return (
    <div className="resource-history">
      <div className="resource-history__legend">
        {series.map((item) => <span className={item.className} key={item.key}><i />{item.label}</span>)}
        <small>{tr("Span {0} · {1} samples · y-axis up to {2}%", spanLabel, visible.length, ceiling)}</small>
      </div>
      <div className="resource-history__canvas">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={tr("CPU, memory and load history of the machine")} preserveAspectRatio="none">
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => <line className="resource-history__grid" key={ratio} x1="0" x2={width} y1={height * ratio} y2={height * ratio} />)}
          {series.flatMap((item) => pointSegments(visible, item.key, width, height, ceiling, firstTimestamp, lastTimestamp).map((points, index) => <polyline className={`resource-history__line ${item.className}`} key={`${item.key}-${index}`} points={points} />))}
          {visible.map((sample, index) => {
            if (sample.reachable) return null;
            const timestamp = new Date(sample.sampledAt).getTime();
            const x = Number.isFinite(timestamp) && lastTimestamp > firstTimestamp
              ? ((timestamp - firstTimestamp) / Math.max(lastTimestamp - firstTimestamp, 1)) * width
              : (index / Math.max(visible.length - 1, 1)) * width;
            return <line className="resource-history__failure" key={`failure-${sample.id}`} x1={x} x2={x} y1="0" y2={height} />;
          })}
        </svg>
      </div>
      <div className="resource-history__axis"><span>{formatDateTime(visible[0]?.sampledAt)}</span><span>{formatDateTime(visible.at(-1)?.sampledAt)}</span></div>
    </div>
  );
}

function GpuMonitor({ gpus, usable, error }: { gpus: ResourceMonitorGpu[]; usable: boolean | null; error?: string | null }) {
  if (!gpus.length) {
    const title = error ? tr("Could not read the GPU state") : usable === false ? tr("No usable NVIDIA GPU detected") : tr("GPU metrics not reported");
    const detail = error || (usable === false ? tr("CPU-only machines still run CPU tasks normally.") : tr("This sample has no confirmed per-GPU metrics."));
    return <div className={`resource-gpu-empty${error ? " is-error" : ""}`}><strong>{title}</strong><span>{detail}</span></div>;
  }
  const utilization = gpus.map((gpu) => gpu.utilizationPercent).filter(finite);
  const usedValues = gpus.map((gpu) => gpu.memoryUsedMb).filter(finite);
  const totalValues = gpus.map((gpu) => gpu.memoryTotalMb).filter(finite);
  const memoryComplete = usedValues.length === gpus.length && totalValues.length === gpus.length;
  const used = memoryComplete ? usedValues.reduce((sum, value) => sum + value, 0) : null;
  const total = memoryComplete ? totalValues.reduce((sum, value) => sum + value, 0) : null;
  const average = utilization.length ? utilization.reduce((sum, value) => sum + value, 0) / utilization.length : null;
  return (
    <details className="resource-gpu-monitor">
      <summary>
        <span><strong>{tr("{0} GPUs", gpus.length)}</strong><small>{tr("Average utilization {0}{1} · {2}", percentageLabel(average), utilization.length !== gpus.length ? tr(" (some GPUs reported)") : "", memoryLabel(used, total))}</small></span>
        <span>{tr("Per GPU")}</span>
      </summary>
      <div className="resource-gpu-list">
        {gpus.map((gpu) => (
          <article key={gpu.index}>
            <header><strong>GPU {gpu.index}</strong><span>{gpu.name || tr("Model not reported")}</span></header>
            {finite(gpu.utilizationPercent) ? <Meter value={gpu.utilizationPercent} label={tr("GPU usage")} detail={`${memoryLabel(gpu.memoryUsedMb, gpu.memoryTotalMb)}${finite(gpu.temperatureC) ? ` · ${Math.round(gpu.temperatureC)}°C` : ""}`} /> : <MissingMetric label={tr("GPU usage")} detail={memoryLabel(gpu.memoryUsedMb, gpu.memoryTotalMb)} />}
          </article>
        ))}
      </div>
    </details>
  );
}

function MissingMetric({ label, detail = tr("This metric was not reported in this sample") }: { label: string; detail?: string }) {
  return <div className="resource-metric-missing"><div><span>{label}</span><strong>—</strong></div><i /><small>{detail}</small></div>;
}

export function ServerResourceMonitor({ monitor }: { monitor: ResourceMonitorViewModel }) {
  const latest = monitor.latestSuccess;
  const metricsAreHistorical = monitor.state !== "fresh";
  return (
    <div className={`resource-monitor resource-monitor--${monitor.state}`}>
      <header className="resource-monitor__header">
        <div>
          <ResourceFreshnessBadge state={monitor.state} />
          <span>{monitor.lastSuccessAt ? tr("Last success {0}", formatRelativeTime(monitor.lastSuccessAt)) : tr("No successful sample yet")}</span>
        </div>
        <small>{tr("Sampled in the background every {0} s · the page reads only the local database", monitor.sampleIntervalSeconds)}</small>
      </header>
      <MonitorNotice monitor={monitor} />
      {latest ? (
        <>
          <div className={`resource-monitor__current${metricsAreHistorical ? " is-historical" : ""}`}>
            {finite(latest.cpuPercent) ? <Meter value={latest.cpuPercent} label={tr("CPU usage")} detail={finite(latest.load1m) ? `load ${latest.load1m.toFixed(2)}` : undefined} /> : <MissingMetric label={tr("CPU usage")} />}
            {finite(latest.memoryPercent) ? <Meter value={latest.memoryPercent} label={tr("Memory usage")} detail={finite(latest.loadRatio) ? `load ratio ${latest.loadRatio.toFixed(2)}` : undefined} /> : <MissingMetric label={tr("Memory usage")} />}
          </div>
          <ResourceHistoryChart samples={monitor.samples} />
          <GpuMonitor gpus={latest.gpus} usable={latest.gpuUsable} error={latest.gpuError} />
        </>
      ) : <div className="resource-monitor__empty"><Clock3 size={20} /><strong>{tr("Waiting for background samples")}</strong><p>{monitorStateDetail(monitor)}</p></div>}
    </div>
  );
}
