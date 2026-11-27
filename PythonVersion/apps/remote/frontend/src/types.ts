export type ViewId = "overview" | "servers" | "tasks" | "runs" | "benchmark" | "analysis" | "settings";

export type ServerStatus = "online" | "probing" | "offline" | "error" | "unknown";

export interface GpuInfo {
  index: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  utilization_percent: number;
  temperature_c?: number | null;
}

export type ServerCheckStatus = "passed" | "warning" | "failed" | "unknown";

export interface ServerEnvironmentCheck {
  key: string;
  label: string;
  status: ServerCheckStatus;
  detail: string;
  blocking: boolean;
}

export interface ServerSystemInfo {
  hostname: string;
  os: string;
  python_version?: string | null;
  conda_env?: string | null;
  cpu_model?: string | null;
  cpu_physical: number;
  cpu_logical: number;
  cpu_percent: number;
  load_1m?: number | null;
  memory_total_gb: number;
  memory_used_gb: number;
  memory_percent: number;
  disk_free_gb: number;
  gpus: GpuInfo[];
  environment_eligible?: boolean | null;
  environment_issues?: string[];
  conda_environment_available?: boolean | null;
  gpu_usable?: boolean | null;
  gpu_error?: string | null;
  disk_writable?: boolean | null;
  load_ratio?: number | null;
  checks?: ServerEnvironmentCheck[];
}

export type ServerResourceMonitorState = "fresh" | "stale" | "failed" | "never" | "paused";

export interface ServerResourceAttempt {
  id: number;
  sampled_at: string;
  reachable: boolean;
  error?: string | null;
}

export interface ServerResourceGpu {
  index: number;
  uuid?: string | null;
  name: string;
  memory_total_mb: number | null;
  memory_used_mb: number | null;
  utilization_percent: number | null;
  temperature_celsius?: number | null;
}

export interface ServerResourceSample {
  aggregation?: { bucket_seconds: number; count: number; start: string; end: string; gpu_values: string; fields: Record<string, { count: number; sum: number; min: number; max: number }> } | Record<string, never>;
  id: number;
  server_id: string;
  sampled_at: string;
  reachable: boolean;
  error?: string | null;
  cpu_physical_cores: number | null;
  cpu_logical_cores: number | null;
  cpu_percent: number | null;
  load_1m: number | null;
  load_5m: number | null;
  load_15m: number | null;
  load_ratio: number | null;
  memory_total_gb: number | null;
  memory_used_gb: number | null;
  memory_percent: number | null;
  gpu_usable?: boolean | null;
  gpu_error?: string | null;
  gpus: ServerResourceGpu[];
}

export interface ServerResourceMonitor {
  state: ServerResourceMonitorState;
  sample_interval_seconds: number;
  stale_after_seconds: number;
  freshness_seconds: number | null;
  pause_reason?: string | null;
  latest_attempt?: ServerResourceAttempt | null;
  last_success?: ServerResourceAttempt | null;
  last_error?: ServerResourceAttempt | null;
  latest_success?: ServerResourceSample | null;
}

export interface ServerResourceLatestResponse {
  sampled_at: string;
  resource_sampling_interval_seconds: number;
  resource_history_retention_hours: number;
  servers: Record<string, ServerResourceMonitor>;
}

export interface ServerResourceHistoryResponse {
  server_id: string;
  resource_monitor: ServerResourceMonitor;
  samples: ServerResourceSample[];
  next_after_id: number | null;
}

export interface RemoteServer {
  id: string;
  target_kind: "ssh" | "local";
  system_managed: boolean;
  name: string;
  host: string;
  port: number;
  username: string;
  ssh_key_path: string;
  remote_root: string;
  conda_env: string;
  enabled: boolean;
  status: ServerStatus;
  last_probe_at?: string | null;
  probe_error?: string | null;
  fingerprint?: string | null;
  system?: ServerSystemInfo | null;
  resource_monitor?: ServerResourceMonitor | null;
}

export interface ServerInput {
  name: string;
  host: string;
  port: number;
  username: string;
  ssh_key_path: string;
  remote_root: string;
  conda_env: string;
  enabled: boolean;
}

export type RunStatus =
  | "queued"
  | "preparing"
  | "syncing"
  | "probing"
  | "running"
  | "retrieving"
  | "cleaning"
  | "cancelling"
  | "succeeded"
  | "failed"
  | "cancelled";

export type RunPhase =
  | "queued"
  | "snapshot"
  | "preflight"
  | "sync"
  | "remote_setup"
  | "execute"
  | "retrieve"
  | "verify"
  | "cleanup"
  | "complete";

export interface RunStageProgress {
  current: number;
  total: number;
  unit: "bytes" | "files";
}

export interface RunStage {
  schema_version: number;
  code: string;
  phase: string;
  state: "running" | "completed" | "failed";
  ordinal: number;
  count: number;
  mode?: "cache" | "delta" | "full" | null;
  message: string;
  progress?: RunStageProgress | null;
  data?: Record<string, unknown>;
}

export interface GitSnapshot {
  commit: string;
  branch?: string | null;
  dirty: boolean;
  snapshot_sha256: string;
  task_sha256: string;
}

export interface RunArtifact {
  path: string;
  kind: "summary" | "trend" | "log" | "manifest" | "other";
  size_bytes: number;
  sha256?: string | null;
  transferred: boolean;
  verified: boolean;
}

export interface RemoteCleanup {
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  attempted_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
}

export interface RunRecovery {
  available: boolean;
  mode?: "transfer" | "launch" | "verify" | "cleanup" | null;
  message?: string | null;
}

export interface RemoteRun {
  id: string;
  status: RunStatus;
  phase: RunPhase;
  stage?: RunStage | null;
  task_path: string;
  task_label?: string | null;
  task_kind?: "batch" | "benchmark" | "plot" | string;
  server_id: string;
  server_name: string;
  target_kind: "ssh" | "local";
  max_parallel_runs: number;
  refresh_interval_seconds: number;
  progress_percent: number;
  completed_units: number;
  total_units: number;
  current_message?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  error_code?: string | null;
  error_message?: string | null;
  git?: GitSnapshot | null;
  artifacts?: RunArtifact[];
  result_candidate_ready?: boolean;
  result_imported?: boolean;
  remote_cleanup?: RemoteCleanup | null;
  recovery?: RunRecovery | null;
  admission_override: boolean;
}

export interface RunDetail extends RemoteRun {
  remote_workspace?: string | null;
  local_result_path?: string | null;
  command?: string | null;
  pid?: number | null;
  environment_report?: EnvironmentReport | null;
}

export interface EnvironmentCheck {
  name: string;
  status: "passed" | "warning" | "failed";
  detail: string;
}

export interface EnvironmentReport {
  eligible: boolean;
  checked_at: string;
  checks: EnvironmentCheck[];
}

export interface RunEvent {
  id: string;
  sequence: number | null;
  timestamp: string;
  stream: "stdout" | "stderr" | "system" | "audit";
  level: "debug" | "info" | "warning" | "error";
  phase?: RunPhase | null;
  message: string;
}

export interface ResourceSample {
  id: number | null;
  timestamp: string;
  cpu_percent: number;
  memory_percent: number;
  load_1m?: number | null;
  gpus: Array<{
    index: number;
    utilization_percent: number;
    memory_used_mb: number;
    memory_total_mb: number;
  }>;
}

export interface TaskFile {
  path: string;
  name: string;
  modified_at?: string | null;
  size_bytes?: number | null;
}

export interface TaskDocument extends TaskFile {
  filename: string;
  content: string;
  sha256: string;
  total_tasks: number;
  total_runs: number;
  total_evaluations: number;
  gpu_requested: boolean;
  warnings: string[];
  entries: TaskInspectionEntry[];
}

export interface TaskDeleteResult {
  deleted: boolean;
  path: string;
  recoverable_path: string;
}

export interface TaskInspectionEntry {
  label: string;
  algorithms: string[];
  problem: string;
  run_count: number;
  fe_per_run: number;
  total_evaluations: number;
}

export interface TaskInspection {
  task_path: string;
  total_tasks: number;
  total_runs: number;
  total_evaluations: number;
  estimated_worker_slots: number;
  algorithms: string[];
  /** Registry names kept only for old tasks, with what they stand for. */
  temporary_algorithms?: Record<string, string>;
  problems: string[];
  entries: TaskInspectionEntry[];
  warnings: string[];
  blocked_reasons: string[];
  hard_blocked_reasons?: string[];
  overrideable_reasons?: string[];
  override_allowed?: boolean;
  admissible: boolean;
}

export interface CreateRunInput {
  display_name?: string;
  task_path?: string;
  task_jsonl?: string;
  server_id: string;
  max_parallel_runs: number;
  refresh_interval_seconds: number;
  admission_override: boolean;
}

export interface DashboardData {
  server_counts: Record<ServerStatus, number>;
  active_run_count: number;
  completed_run_count: number;
  failed_run_count: number;
  transferred_bytes: number;
  servers: RemoteServer[];
  recent_runs: RemoteRun[];
  active_runs: RemoteRun[];
  audit_events: AuditEvent[];
}

export interface HealthData {
  status: "ok" | "degraded";
  service: string;
  version?: string | null;
  database?: "ok" | "error" | "unknown";
  workspace?: string | null;
}

export interface AuditEvent {
  id: string;
  timestamp: string;
  action: string;
  actor: string;
  target: string;
  detail: string;
  outcome: "success" | "warning" | "failure";
}

export interface RemoteSettings {
  workspace_path: string;
  default_ssh_key_path: string;
  default_remote_root: string;
  default_poll_interval_seconds: number;
  default_max_parallel_runs: number;
  probe_timeout_seconds: number;
  resource_sampling_interval_seconds: number;
  resource_history_retention_hours: number;
  require_clean_git: boolean;
  verify_artifact_checksums: boolean;
  delete_remote_results_after_transfer: boolean;
}

export interface ListResponse<T> {
  items: T[];
}

export interface ResultCatalogItem {
  result_id: string;
  run_id: string;
  display_name: string;
  task_name: string;
  task_path?: string;
  source_kind: string;
  source_run_id?: string | null;
  source_metadata: Record<string, unknown>;
  imported_at: string;
  created_at?: string;
  finished_at?: string | null;
  bundle_id: string;
  bundle_status: string;
  task_count: number;
  runset_count: number;
  expected_runs: number;
  completed_runs: number;
  trend_metric_count: number;
  imported_runset_count: number;
  preview_available: boolean;
}

export interface ResultCandidate {
  candidate_id: string;
  run_id: string;
  display_name: string;
  task_name: string;
  task_path: string;
  task_kind: string;
  server_id: string;
  server_name: string;
  target_kind: "ssh" | "local";
  created_at: string;
  completed_at?: string | null;
  completed_units: number;
  total_units: number;
  bundle_relative_dir: string;
  available: boolean;
  imported: boolean;
  imported_result_id?: string | null;
}

export interface ResultBundleImportResponse {
  result: ResultLibraryEntry;
  catalog_item: ResultCatalogItem | null;
  preview: RunResultsPreview;
}

export interface ResultArchiveUploadResponse {
  count_unit: "runset";
  imported_count: number;
  skipped_count: number;
  failed_count: number;
  total_count: number;
  imported: ResultBundleImportResponse[];
  imported_runsets: Array<{
    display_name: string;
    runset_uid: string;
    result_id: string;
  }>;
  skipped: Array<{
    display_name: string;
    runset_uid: string;
    reason: "duplicate";
    duplicate_of_result_id?: string | null;
    duplicate_of_display_name: string;
    duplicate_of_runset_uid: string;
  }>;
  failed: Array<{
    display_name: string;
    runset_uid?: string | null;
    code: string;
    message: string;
  }>;
}

export interface ResultLibraryEntry {
  id: string;
  result_id: string;
  run_id: string;
  display_name: string;
  task_name: string;
  source_kind: string;
  source_run_id?: string | null;
  source_metadata: Record<string, unknown>;
  managed_relative_dir: string;
  validation: ResultValidation;
  imported_at: string;
}

export interface ResultValidation {
  bundle_id: string;
  status: string;
  task_count: number;
  runset_count: number;
  expected_runs: number;
  completed_runs: number;
  trend_metric_count: number;
  [key: string]: unknown;
}

export interface ResultTaskEntry {
  task_id: number;
  task_uid: string;
  label: string;
  source_task_file?: string | null;
  source_line?: number | null;
  task_config_hash: string;
  problem_name: string;
  run_number: number;
  problem_params: Record<string, unknown>;
  status: string;
  created_at: string;
  completed_at?: string | null;
  error_message?: string | null;
}

export interface ScalarMetricSummary {
  name: string;
  mean: number | null;
  median: number | null;
  stderr: number | null;
  count: number;
  direction: "lower" | "higher" | string;
}

export interface TrendMetricSummary {
  name: string;
  axis: string;
  sample_count: number;
}

export interface ResultRunset {
  runset_id: number;
  runset_uid: string;
  task_id: number;
  task_uid: string;
  label: string;
  algorithm_name: string;
  problem_name: string;
  algorithm_params: Record<string, unknown>;
  problem_params: Record<string, unknown>;
  expected_runs: number;
  completed_runs: number;
  status: string;
  source_task_file?: string | null;
  source_line?: number | null;
  task_config_hash: string;
  scalar_metrics: ScalarMetricSummary[];
  trend_metrics: TrendMetricSummary[];
  pairing_signature: string;
  sa_import_id?: string | null;
}

export interface RunResultsPreview {
  result_id: string;
  run_id: string;
  display_name?: string;
  task_name: string;
  task_path: string;
  source_kind?: string;
  source_run_id?: string | null;
  source_metadata?: Record<string, unknown>;
  run_status: string;
  exit_code?: number | null;
  snapshot_sha256?: string | null;
  task_sha256?: string | null;
  source_fingerprint: string;
  validation: ResultValidation;
  algorithms: string[];
  problems: string[];
  scalar_metrics: string[];
  trend_metrics: string[];
  tasks: ResultTaskEntry[];
  runsets: ResultRunset[];
}

export interface ResultRunSample {
  run_uid: string;
  run_index: number;
  problem_seed: number;
  algorithm_seed: number;
  fe: number;
  runtime_seconds?: number | null;
  status: string;
  iteration: number;
  compute: Record<string, unknown>;
  metrics: Record<string, number | null>;
  created_at: string;
}

export interface RunsetSamplesPreview {
  result_id: string;
  run_id: string;
  runset_uid: string;
  label: string;
  algorithm_name: string;
  problem_name: string;
  problem_params: Record<string, unknown>;
  algorithm_params: Record<string, unknown>;
  expected_runs: number;
  completed_runs: number;
  status: string;
  offset: number;
  limit: number;
  total: number;
  runs: ResultRunSample[];
}

export interface RunsetTrendPreview {
  result_id: string;
  run_id: string;
  runset_uid: string;
  metric_name: string;
  axis: string;
  total: number;
  offset: number;
  limit: number;
  key: number[];
  mean: Array<number | null>;
  stderr: Array<number | null>;
  count: number[];
}

export interface SAImport {
  id: string;
  result_id: string;
  run_id: string;
  runset_uid: string;
  bundle_id: string;
  task_name: string;
  label: string;
  algorithm_name: string;
  problem_name: string;
  problem_params: Record<string, unknown>;
  algorithm_params: Record<string, unknown>;
  scalar_metrics: ScalarMetricSummary[];
  trend_metrics: TrendMetricSummary[];
  expected_runs: number;
  completed_runs: number;
  source_fingerprint: string;
  pairing_signature: string;
  snapshot_sha256: string;
  task_sha256: string;
  imported_at: string;
  source_available?: boolean;
  source_status?: string;
}

export interface SAImportResult {
  imported: number;
  already_imported: number;
  skipped: number;
  skipped_details: Array<{ runset_uid: string; label: string; reason: string }>;
  items: SAImport[];
  duplicate_warning?: boolean;
}

export interface SAQuickFilterResult {
  template_import_id: string;
  added: number;
  already: number;
  skipped: number;
  items: SAImport[];
  already_items?: SAImport[];
  skipped_items?: Array<Record<string, unknown>>;
}

export type SAMethod = "friedman" | "signed-rank" | "rank-sum";
export type SAOutputFormat = "sqlite" | "csv" | "xlsx" | "latex";

export interface SAMatchResult {
  method: SAMethod;
  minimum: number;
  imports: SAImport[];
  import_ids: string[];
  metrics: string[];
  compatible: boolean;
  message: string;
  warnings: string[];
}

export interface SASourceRunset {
  import_id: string;
  result_id?: string;
  run_id: string;
  runset_uid: string;
  task_name: string;
  label: string;
  algorithm_name: string;
  problem_name: string;
  problem_params: Record<string, unknown>;
  algorithm_params: Record<string, unknown>;
  expected_runs?: number;
  completed_runs: number;
  status?: string;
  scalar_metrics?: ScalarMetricSummary[];
  trend_metrics?: TrendMetricSummary[];
  display_label?: string;
  label_dimension?: SALabelDimension;
  source_fingerprint: string;
  snapshot_sha256: string;
  task_sha256: string;
}

export interface SALabelDimension {
  kind: "algorithm" | "algorithm_parameter";
  parameter?: string | null;
  title: string;
}

export interface SAResultTable {
  metric_name: string;
  columns: string[];
  rows: Array<Array<string | number | null>>;
  runset_uids?: string[];
  [key: string]: unknown;
}

export interface SAResultPayload {
  method: SAMethod;
  metrics: string[];
  alpha: number;
  runset_ids: Array<string | number>;
  runset_uids: string[];
  analyses: Array<Record<string, unknown>>;
  tables: SAResultTable[];
  created_at: string;
  analysis_id?: string;
  name?: string;
  selected_runsets?: SASourceRunset[];
  label_dimension?: SALabelDimension;
  display_labels?: string[];
}

export interface SAAnalysis {
  id: string;
  name: string;
  status: "running" | "completed" | "failed" | string;
  method: SAMethod;
  alpha: number;
  metric_names: string[];
  source_imports: SASourceRunset[];
  result: SAResultPayload | Record<string, never>;
  output_dir?: string | null;
  output_files: Record<string, string>;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface SAAnalysisInput {
  name: string;
  method: SAMethod;
  import_ids: string[];
  metrics?: string[];
  alpha: number;
  formats?: SAOutputFormat[];
  label_parameter?: string;
}
