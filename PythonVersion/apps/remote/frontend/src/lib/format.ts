import type { RunPhase, RunStatus, ServerStatus } from "../types";
import { tr, LOCALE } from "./i18n";

export const formatDateTime = (value?: string | null): string => {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(LOCALE, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
};

export const formatRelativeTime = (value?: string | null): string => {
  if (!value) return tr("never");
  const distance = Date.now() - new Date(value).getTime();
  if (distance < 60_000) return tr("just now");
  if (distance < 3_600_000) return Math.floor(distance / 60_000) + tr(" min ago");
  if (distance < 86_400_000) return Math.floor(distance / 3_600_000) + tr(" h ago");
  return Math.floor(distance / 86_400_000) + tr(" d ago");
};

export const formatBytes = (value: number): string => {
  if (value < 1024) return value + " B";
  if (value < 1024 ** 2) return (value / 1024).toFixed(1) + " KB";
  if (value < 1024 ** 3) return (value / 1024 ** 2).toFixed(1) + " MB";
  return (value / 1024 ** 3).toFixed(2) + " GB";
};

export const formatInteger = (value: number): string => new Intl.NumberFormat(LOCALE).format(value);

export const runStatusLabel: Record<RunStatus, string> = {
  queued: tr("Queued"),
  preparing: tr("Preparing"),
  syncing: tr("Code sync"),
  probing: tr("Environment check"),
  running: tr("Running"),
  retrieving: tr("Collecting results"),
  cleaning: tr("Cleaning the execution folder"),
  cancelling: tr("Cancelling"),
  succeeded: tr("Completed"),
  failed: tr("Failed"),
  cancelled: tr("Cancelled"),
};

export const phaseLabel: Record<RunPhase, string> = {
  queued: tr("Queued for execution"),
  snapshot: tr("Freezing the code snapshot"),
  preflight: tr("Checking the environment"),
  sync: tr("Syncing changes"),
  remote_setup: tr("Preparing the execution workspace"),
  execute: tr("Running the experiment"),
  retrieve: tr("Collecting all results"),
  verify: tr("Validating artifacts"),
  cleanup: tr("Cleaning temporary results on the executing machine"),
  complete: tr("Recording completion"),
};

export const serverStatusLabel: Record<ServerStatus, string> = {
  online: tr("Connected"),
  probing: tr("Checking"),
  offline: tr("Offline"),
  error: tr("Failed"),
  unknown: tr("Not checked"),
};

export const terminalRunStatuses: RunStatus[] = ["succeeded", "failed", "cancelled"];

export const isTerminalRun = (status: RunStatus): boolean => terminalRunStatuses.includes(status);

export const shortSha = (value?: string | null): string => value ? value.slice(0, 10) : "—";
