#!/bin/sh
set -eu

key_path=${EDOLAB_RELAY_SSH_KEY:-}
case "$key_path" in
  /*) ;;
  *)
    echo "EDOLAB Relay: EDOLAB_RELAY_SSH_KEY must be an absolute host key path." >&2
    exit 2
    ;;
esac
if [ ! -s "$key_path" ] || [ ! -r "$key_path" ]; then
  echo "EDOLAB Relay: the read-only SSH key mount is missing or unreadable." >&2
  exit 2
fi

# The application itself remains loopback-only.  socat exposes it only to the
# container interface; Compose publishes that interface on host 127.0.0.1.
socat TCP-LISTEN:8766,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:8767 &

exec python /opt/edolab/startup_remote.py \
  --host 127.0.0.1 \
  --port 8767 \
  --data-root /workspace/results/remote_control \
  --no-open-browser \
  "$@"
