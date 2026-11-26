import { useCallback, useEffect, useRef, useState } from "react";
import { createPolling } from "../lib/polling";
import { tr } from "../lib/i18n";

interface PollingState<T> {
  data: T | null;
  loading: boolean;
  refreshing: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

export const usePolling = <T,>(
  loader: () => Promise<T>,
  intervalSeconds: number,
  enabled = true,
): PollingState<T> => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const refreshCurrent = useRef<(() => Promise<void>) | null>(null);
  const refresh = useCallback(() => refreshCurrent.current?.() ?? Promise.resolve(), []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      setRefreshing(false);
      return;
    }
    let active = true;
    const load = async (): Promise<void> => {
      setRefreshing(true);
      await Promise.resolve().then(loader).then((next) => {
        if (active) {
          setData(next);
          setError(null);
        }
      }).catch((nextError: unknown) => {
        if (active) setError(nextError instanceof Error ? nextError : new Error(tr("Read failed")));
      }).finally(() => {
        if (active) {
          setLoading(false);
          setRefreshing(false);
        }
      });
    };
    const polling = createPolling(load, () => intervalSeconds * 1000);
    refreshCurrent.current = polling.refresh;
    return () => {
      active = false;
      refreshCurrent.current = null;
      polling.dispose();
    };
  }, [enabled, intervalSeconds, loader]);

  return { data, loading, refreshing, error, refresh };
};
