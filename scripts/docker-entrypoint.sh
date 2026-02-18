#!/bin/bash
# If DEV_VENV=1, use the host-mounted venv at /app/backend/venv instead of /opt/venv.
# This lets you pip install into the host venv and have changes reflected immediately
# without rebuilding the image.
if [ "${DEV_VENV:-0}" = "1" ]; then
    export PATH="/app/backend/venv/bin:$PATH"
else
    export PATH="/opt/venv/bin:$PATH"
fi

exec "$@"
