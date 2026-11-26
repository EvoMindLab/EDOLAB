import { useEffect, useState, type ReactNode } from "react";
import {
  BarChart3,
  FileCode2,
  LayoutDashboard,
  PanelLeft,
  Play,
  Plus,
  RefreshCw,
  Server,
  Settings2,
  Waypoints,
} from "lucide-react";
import type { HealthData, ViewId } from "../types";
import { THEME_OPTIONS, type ThemeId } from "../lib/theme";
import { Button } from "./ui";
import { LanguageSwitch } from "./LanguageSwitch";
import { tr } from "../lib/i18n";

interface NavItem {
  id: ViewId;
  label: string;
  icon: typeof LayoutDashboard;
}

const navGroups: Array<{ label: string; items: NavItem[] }> = [
  {
    label: tr("Experiments"),
    items: [
      { id: "overview", label: tr("Overview"), icon: LayoutDashboard },
      { id: "servers", label: tr("Machines"), icon: Server },
      { id: "tasks", label: tr("Tasks"), icon: FileCode2 },
      { id: "runs", label: tr("Runs"), icon: Play },
    ],
  },
  {
    label: tr("Analysis"),
    items: [
      { id: "analysis", label: tr("Results"), icon: BarChart3 },
      { id: "benchmark", label: tr("Problems and NBN"), icon: Waypoints },
    ],
  },
];

const SIDEBAR_STORAGE_KEY = "edolab.relay.sidebarCollapsed";

const readCollapsed = (): boolean => {
  try {
    return window.localStorage.getItem(SIDEBAR_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
};

export function AppShell({
  view,
  onViewChange,
  onNewRun,
  health,
  healthError,
  refreshing,
  onRefresh,
  theme,
  onThemeChange,
  children,
}: {
  view: ViewId;
  onViewChange: (view: ViewId) => void;
  onNewRun: () => void;
  health: HealthData | null;
  healthError: Error | null;
  refreshing: boolean;
  onRefresh: () => void;
  theme: ThemeId;
  onThemeChange: (theme: ThemeId) => void;
  children: ReactNode;
}) {
  const [collapsed, setCollapsed] = useState<boolean>(readCollapsed);
  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_STORAGE_KEY, collapsed ? "1" : "0");
    } catch {
      // ignore storage quota/privacy failures
    }
  }, [collapsed]);

  const serviceHealthy = health?.status === "ok" && !healthError;
  const serviceLabel = healthError ? tr("Service unreachable") : serviceHealthy ? tr("Service running") : tr("Service degraded");
  const serviceClass = healthError ? "is-error" : serviceHealthy ? "is-ok" : "is-warn";

  return (
    <div className={"app-shell" + (collapsed ? " is-collapsed" : "")}>
      <aside className="sidebar" aria-label={tr("Main navigation")}>
        <div className="sidebar__top">
          <button
            type="button"
            className="sidebar__brand"
            onClick={() => onViewChange("overview")}
            aria-label={tr("Back to overview")}
          >
            <strong>EDOLAB</strong>
            <span>Relay</span>
          </button>
          <button
            type="button"
            className="sidebar__toggle"
            aria-label={collapsed ? tr("Expand sidebar") : tr("Collapse sidebar")}
            title={collapsed ? tr("Expand sidebar") : tr("Collapse sidebar")}
            onClick={() => setCollapsed((current) => !current)}
          >
            <PanelLeft size={16} aria-hidden="true" />
          </button>
        </div>

        <nav className="sidebar__nav">
          {navGroups.map((group) => (
            <div className="sidebar__group" key={group.label}>
              <span className="sidebar__group-label">{group.label}</span>
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.id}
                    type="button"
                    className={"sidebar__item" + (view === item.id ? " is-active" : "")}
                    title={collapsed ? item.label : undefined}
                    aria-current={view === item.id ? "page" : undefined}
                    onClick={() => onViewChange(item.id)}
                  >
                    <Icon size={16} strokeWidth={1.75} aria-hidden="true" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar__footer">
          <button
            type="button"
            className={"sidebar__item" + (view === "settings" ? " is-active" : "")}
            title={collapsed ? tr("Settings and audit") : undefined}
            aria-current={view === "settings" ? "page" : undefined}
            onClick={() => onViewChange("settings")}
          >
            <Settings2 size={16} strokeWidth={1.75} aria-hidden="true" />
            <span>{tr("Settings and audit")}</span>
          </button>
          <div
            className={"sidebar__status " + serviceClass}
            title={healthError?.message ?? health?.workspace ?? serviceLabel}
          >
            <i aria-hidden="true" />
            <span>{serviceLabel}</span>
          </div>
        </div>
      </aside>

      <header className="topbar">
        <button
          type="button"
          className="topbar__brand"
          onClick={() => onViewChange("overview")}
          aria-label={tr("Back to overview")}
        >
          <strong>EDOLAB</strong>
          <span>Relay</span>
        </button>
        <div className="topbar__actions">
          <LanguageSwitch />
          <div className="theme-switcher" role="radiogroup" aria-label={tr("Color theme")}>
            {THEME_OPTIONS.map((option) => {
              const active = option.id === theme;
              return (
                <button
                  key={option.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  title={tr("{0} · {1}: {2}", option.index, option.label, option.description)}
                  className={"theme-option" + (active ? " is-active" : "")}
                  onClick={() => onThemeChange(option.id)}
                >
                  <i
                    className="theme-option__swatch"
                    aria-hidden="true"
                    style={{
                      background: `linear-gradient(135deg, ${option.surfaceSwatch} 48%, ${option.accentSwatch} 52%)`,
                    }}
                  />
                  <span>{option.label}</span>
                </button>
              );
            })}
          </div>
          <button
            type="button"
            className="topbar__icon-btn"
            onClick={onRefresh}
            aria-label={tr("Refresh now")}
            title={tr("Refresh now")}
          >
            <RefreshCw size={15} strokeWidth={1.8} className={refreshing ? "is-spinning" : ""} />
          </button>
          <Button tone="primary" onClick={onNewRun} icon={<Plus size={15} strokeWidth={2.2} />}>{tr("New run")}</Button>
        </div>
      </header>

      <main className="workspace-scroll">{children}</main>
    </div>
  );
}
