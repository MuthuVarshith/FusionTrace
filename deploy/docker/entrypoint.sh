#!/bin/sh
# FusionTrace container entrypoint.
#
#   serve (default)   fetch missing detector weights, then run the API on $PORT
#   anything else     run that command instead, e.g.
#                     docker compose run --rm fusiontrace python scripts/download_models.py --all
#
# Weight download is idempotent: bundles already in /app/models (a volume) are
# skipped without network access. Set FUSIONTRACE_DOWNLOAD_MODELS=0 to skip it
# entirely, e.g. when weights are mounted read-only.
set -eu

if [ "${1:-serve}" != "serve" ]; then
    exec "$@"
fi

if [ "${FUSIONTRACE_DOWNLOAD_MODELS:-1}" != "0" ]; then
    if ! python scripts/download_models.py; then
        # Start anyway: detectors without weights report "available": false on
        # /health and scans fail with a clear "model not available" message.
        echo "entrypoint: some detector weights are missing; starting in degraded mode (see /health)" >&2
    fi
fi

# One worker: detectors are process-wide singletons and scans run as in-process
# background tasks, so extra workers would multiply model memory.
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
