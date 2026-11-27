import { useEffect, type ButtonHTMLAttributes, type PropsWithChildren, type ReactNode } from "react";
import { AlertTriangle, Check, CircleAlert, Info, X } from "lucide-react";
import type { RunStatus, ServerStatus } from "../types";
import { runStatusLabel, serverStatusLabel } from "../lib/format";
import { tr } from "../lib/i18n";

type ButtonTone = "primary" | "secondary" | "quiet" | "danger";
export type BadgeTone = "neutral" | "info" | "success" | "warning" | "danger";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: ButtonTone;
  icon?: ReactNode;
}

export function Button({ tone = "secondary", icon, children, className = "", ...props }: ButtonProps) {
  return (
    <button className={("button button--" + tone + " " + className).trim()} {...props}>
      {icon}
      {children}
    </button>
  );
}

export function IconButton({ children, className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={("icon-button " + className).trim()} {...props}>{children}</button>;
}

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: ReactNode }) {
  return (
    <header className="page-heading">
      <div className="page-heading__copy">
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {actions ? <div className="page-heading__actions">{actions}</div> : null}
    </header>
  );
}

export function Badge({
  tone = "neutral",
  dot = false,
  pulse = false,
  icon,
  className = "",
  title,
  children,
}: PropsWithChildren<{
  tone?: BadgeTone;
  dot?: boolean;
  pulse?: boolean;
  icon?: ReactNode;
  className?: string;
  title?: string;
}>) {
  return (
    <span
      className={("badge badge--" + tone + (pulse ? " is-pulsing" : "") + " " + className).trim()}
      title={title}
    >
      {dot ? <i aria-hidden="true" /> : icon}
      {children}
    </span>
  );
}

export interface TabItem<T extends string> {
  value: T;
  label: ReactNode;
  count?: ReactNode;
  icon?: ReactNode;
  title?: string;
  disabled?: boolean;
  id?: string;
  controls?: string;
}

export function Tabs<T extends string>({
  items,
  value,
  onChange,
  variant = "underline",
  ariaLabel,
  className = "",
}: {
  items: Array<TabItem<T>>;
  value: T;
  onChange: (value: T) => void;
  variant?: "underline" | "segmented";
  ariaLabel: string;
  className?: string;
}) {
  return (
    <div className={("ui-tabs ui-tabs--" + variant + " " + className).trim()} role="tablist" aria-label={ariaLabel}>
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={item.value}
            id={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            aria-controls={item.controls}
            className={active ? "is-active" : ""}
            title={item.title}
            disabled={item.disabled}
            onClick={() => onChange(item.value)}
          >
            {item.icon}
            <span>{item.label}</span>
            {item.count !== undefined ? <small>{item.count}</small> : null}
          </button>
        );
      })}
    </div>
  );
}

export function CodeBlock({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <pre className={("code-block " + className).trim()}>{children}</pre>;
}

const runTone: Record<RunStatus, BadgeTone> = {
  queued: "neutral",
  preparing: "info",
  syncing: "info",
  probing: "info",
  running: "info",
  retrieving: "info",
  cleaning: "info",
  cancelling: "warning",
  succeeded: "success",
  failed: "danger",
  cancelled: "neutral",
};

const serverTone: Record<ServerStatus, BadgeTone> = {
  online: "success",
  probing: "info",
  offline: "neutral",
  error: "danger",
  unknown: "warning",
};

export function RunStatusPill({ status }: { status: RunStatus }) {
  const tone = runTone[status];
  return <Badge tone={tone} dot pulse={tone === "info"}>{runStatusLabel[status]}</Badge>;
}

export function ServerStatusPill({ status }: { status: ServerStatus }) {
  const tone = serverTone[status];
  return <Badge tone={tone} dot pulse={tone === "info"}>{serverStatusLabel[status]}</Badge>;
}

export function Meter({ value, label, detail }: { value: number; label: string; detail?: string }) {
  const safeValue = Math.max(0, Math.min(100, value));
  const tone = safeValue < 40 ? "healthy" : safeValue < 80 ? "warning" : "danger";
  return (
    <div className={"meter meter--" + tone}>
      <div className="meter__label"><span>{label}</span><strong>{Math.round(safeValue)}%</strong></div>
      <div className="meter__track" aria-label={label + " " + Math.round(safeValue) + "%"} role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={safeValue}>
        <span style={{ width: safeValue + "%" }} />
      </div>
      <small className={detail ? "" : "is-placeholder"}>{detail ?? "\u00a0"}</small>
    </div>
  );
}

export function Notice({ tone = "info", children }: PropsWithChildren<{ tone?: "info" | "warning" | "danger" | "success" }>) {
  const icons = { info: <Info size={16} />, warning: <AlertTriangle size={16} />, danger: <CircleAlert size={16} />, success: <Check size={16} /> };
  return <div className={"notice notice--" + tone}>{icons[tone]}<div>{children}</div></div>;
}

export function Dialog({ open, title, description, onClose, children, footer, wide = false }: PropsWithChildren<{ open: boolean; title: string; description?: string; onClose: () => void; footer?: ReactNode; wide?: boolean }>) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, open]);

  if (!open) return null;
  return (
    <div className="dialog-layer" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <section className={"dialog" + (wide ? " dialog--wide" : "")} role="dialog" aria-modal="true" aria-labelledby="dialog-title">
        <header className="dialog__header">
          <div><h2 id="dialog-title">{title}</h2>{description ? <p>{description}</p> : null}</div>
          <IconButton aria-label={tr("Close")} onClick={onClose}><X size={18} /></IconButton>
        </header>
        <div className="dialog__body">{children}</div>
        {footer ? <footer className="dialog__footer">{footer}</footer> : null}
      </section>
    </div>
  );
}

export function ConfirmDialog({ open, title, message, confirmLabel, destructive = false, busy = false, onConfirm, onClose }: { open: boolean; title: string; message: string; confirmLabel: string; destructive?: boolean; busy?: boolean; onConfirm: () => void; onClose: () => void }) {
  return (
    <Dialog open={open} title={title} onClose={onClose} footer={
      <><Button onClick={onClose} disabled={busy}>{tr("Back")}</Button><Button tone={destructive ? "danger" : "primary"} onClick={onConfirm} disabled={busy}>{busy ? tr("Working…") : confirmLabel}</Button></>
    }>
      <p className="confirm-copy">{message}</p>
    </Dialog>
  );
}

export function LoadingBlock({ label = tr("Loading…") }: { label?: string }) {
  return <div className="loading-block"><span /><p>{label}</p></div>;
}

export function EmptyState({ title, description, action }: { title: string; description: string; action?: ReactNode }) {
  return <div className="empty-state"><h3>{title}</h3><p>{description}</p>{action}</div>;
}
