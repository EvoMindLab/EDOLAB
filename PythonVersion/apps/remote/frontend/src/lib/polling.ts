export interface PollingEnvironment {
  hidden: () => boolean;
  subscribe: (listener: () => void) => () => void;
  later: (callback: () => void, delay: number) => unknown;
  cancel: (timer: unknown) => void;
}

const browserEnvironment = (): PollingEnvironment => ({
  hidden: () => document.hidden,
  subscribe: (listener) => {
    document.addEventListener("visibilitychange", listener);
    return () => document.removeEventListener("visibilitychange", listener);
  },
  later: (callback, delay) => window.setTimeout(callback, delay),
  cancel: (timer) => window.clearTimeout(timer as number),
});

// Completion-based scheduling with one request in flight. Consumers own error
// display and guard state updates after disposal; this helper owns scheduling.
export function createPolling(task: () => Promise<void>, delay: () => number,
  environment: PollingEnvironment = browserEnvironment()) {
  let active = true;
  let pending: Promise<void> | null = null;
  let timer: unknown = null;
  const clear = () => { if (timer !== null) environment.cancel(timer); timer = null; };
  const schedule = () => {
    clear();
    if (active && !environment.hidden()) timer = environment.later(() => { void refresh(); }, Math.max(1000, delay()));
  };
  const refresh = (): Promise<void> => {
    if (!active) return Promise.resolve();
    if (pending) return pending;
    clear();
    pending = Promise.resolve().then(() => { if (active) return task(); }).finally(() => { pending = null; schedule(); });
    return pending;
  };
  const unsubscribe = environment.subscribe(() => {
    clear();
    if (!environment.hidden()) void refresh();
  });
  if (!environment.hidden()) void refresh();
  return { refresh, dispose: () => { active = false; clear(); unsubscribe(); } };
}
