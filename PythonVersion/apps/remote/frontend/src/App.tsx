import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { Toaster } from "sonner";
import { api } from "./api/client";
import { AppShell } from "./components/AppShell";
import { LoadingBlock } from "./components/ui";
import { DashboardPage } from "./features/dashboard/DashboardPage";
import { RunsPage } from "./features/runs/RunsPage";
import { ServersPage } from "./features/servers/ServersPage";
import { SettingsPage } from "./features/settings/SettingsPage";
import { TaskLibraryPage } from "./features/tasks";
import { usePolling } from "./hooks/usePolling";
import { applyTheme, readSavedTheme, type ThemeId } from "./lib/theme";

// Heavy pages load on first visit: the landscape views bring three.js and analysis is large on its own.
const AnalysisPage = lazy(() => import("./features/analysis/AnalysisPage").then(m => ({ default: m.AnalysisPage })));
const BenchmarkPage = lazy(() => import("./features/benchmark").then(m => ({ default: m.BenchmarkPage })));
import type { RemoteRun, ViewId } from "./types";
import { tr } from "./lib/i18n";

const views: ViewId[] = ["overview", "servers", "tasks", "runs", "benchmark", "analysis", "settings"];
type AnalysisTab = "candidates" | "results" | "workspace" | "history";

const analysisEntryTabFromHash = (): AnalysisTab => {
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash === "results" || hash === "workspace" || hash === "history") return hash;
  if (hash === "plots") return "results";
  return "candidates";
};

const viewFromHash = (): ViewId => {
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash === "plots" || hash === "results" || hash === "workspace" || hash === "history") return "analysis";
  return views.includes(hash as ViewId) ? hash as ViewId : "overview";
};

export default function App() {
  const [theme, setTheme] = useState<ThemeId>(readSavedTheme);
  const [view, setView] = useState<ViewId>(viewFromHash);
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>(analysisEntryTabFromHash);
  const [openPlotComposerSignal, setOpenPlotComposerSignal] = useState(
    () => window.location.hash.replace(/^#\/?/, "") === "plots" ? 1 : 0,
  );
  const [runPollSeconds, setRunPollSeconds] = useState(5);
  const [resourcePollSeconds, setResourcePollSeconds] = useState(3);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [openCreateSignal, setOpenCreateSignal] = useState(0);
  const [prefillTaskPath, setPrefillTaskPath] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const healthLoader = useCallback(() => api.health(), [refreshNonce]);
  const health = usePolling(healthLoader, Math.max(runPollSeconds, 5));

  useEffect(() => {
    void api.settings.get().then((settings) => {
      setRunPollSeconds(settings.default_poll_interval_seconds);
      setResourcePollSeconds(settings.resource_sampling_interval_seconds);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    const onHashChange = () => {
      setView(viewFromHash());
      setAnalysisTab(analysisEntryTabFromHash());
      if (window.location.hash.replace(/^#\/?/, "") === "plots") {
        setOpenPlotComposerSignal((current) => current + 1);
      }
    };
    window.addEventListener("hashchange", onHashChange);
    window.addEventListener("popstate", onHashChange);
    return () => { window.removeEventListener("hashchange", onHashChange); window.removeEventListener("popstate", onHashChange); };
  }, []);

  const changeView = (next: ViewId) => {
    if (next === "analysis") setAnalysisTab("candidates");
    setView(next);
    window.history.replaceState(null, "", "/#/" + next);
  };

  const newRun = () => {
    setPrefillTaskPath(null);
    changeView("runs");
    setOpenCreateSignal((current) => current + 1);
  };

  const runTask = (taskPath: string) => {
    setPrefillTaskPath(taskPath);
    changeView("runs");
    setOpenCreateSignal((current) => current + 1);
  };

  const openRun = (run: RemoteRun) => {
    changeView("runs");
    setSelectedRunId(run.id);
  };

  const refreshAll = () => {
    setRefreshNonce((current) => current + 1);
    void health.refresh();
  };

  const changeTheme = (next: ThemeId) => {
    setTheme(next);
    applyTheme(next);
  };

  return (
    <>
      <AppShell view={view} onViewChange={changeView} onNewRun={newRun} health={health.data} healthError={health.error} refreshing={health.refreshing} onRefresh={refreshAll} theme={theme} onThemeChange={changeTheme}>
        {view === "overview" ? <DashboardPage pollSeconds={resourcePollSeconds} refreshNonce={refreshNonce} onNewRun={newRun} onViewChange={changeView} onOpenRun={openRun} /> : null}
        {view === "servers" ? <ServersPage pollSeconds={resourcePollSeconds} refreshNonce={refreshNonce} /> : null}
        {view === "tasks" ? <TaskLibraryPage refreshNonce={refreshNonce} onRunTask={runTask} /> : null}
        {view === "runs" ? <RunsPage pollSeconds={runPollSeconds} refreshNonce={refreshNonce} openCreateSignal={openCreateSignal} initialTaskPath={prefillTaskPath} selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} /> : null}
        <Suspense fallback={<LoadingBlock label={tr("Loading page…")} />}>
          {view === "benchmark" ? <BenchmarkPage api={api.benchmarks} refreshNonce={refreshNonce} /> : null}
          {view === "analysis" ? <AnalysisPage refreshNonce={refreshNonce} requestedTab={analysisTab} onRequestedTabChange={setAnalysisTab} openPlotComposerSignal={openPlotComposerSignal} /> : null}
        </Suspense>
        {view === "settings" ? <SettingsPage pollSeconds={resourcePollSeconds} refreshNonce={refreshNonce} onPollSecondsChange={setRunPollSeconds} onResourcePollSecondsChange={setResourcePollSeconds} /> : null}
      </AppShell>
      <Toaster theme="light" position="bottom-right" richColors closeButton toastOptions={{ className: "relay-toast" }} />
    </>
  );
}
