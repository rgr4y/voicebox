#!/bin/bash
# Docker entrypoint — selects the Python venv based on DEV_VENV env var.
#
# DEV_VENV=1: use the host-mounted venv at /app/backend/venv
#             (volume-mounted from the host for fast iteration without rebuilds)
# DEV_VENV=0 (default): use /opt/venv built into the image at build time

echo "[entrypoint] DEV_VENV=${DEV_VENV:-0}"

if [ "${DEV_VENV:-0}" = "1" ]; then
    VENV=/app/backend/venv
    echo "[entrypoint] Using host venv: $VENV"
    if [ ! -f "$VENV/pyvenv.cfg" ]; then
        echo "[entrypoint] ERROR: host venv not found at $VENV"
        echo "[entrypoint]   Is the volume mounted and venv created on the Linux host?"
        echo "[entrypoint]   Run on Linux host: cd voicebox && make setup-python-linux-cuda"
        exit 1
    fi
    # Verify the python symlink resolves inside the container (not a macOS/pyenv path)
    if ! "$VENV/bin/python3" --version >/dev/null 2>&1; then
        echo "[entrypoint] ERROR: venv python3 symlink is broken inside the container"
        echo "[entrypoint]   The venv was likely created on macOS (pyenv symlinks don't resolve here)"
        echo "[entrypoint]   Recreate it on the Linux host: cd voicebox && rm -rf backend/venv && make setup-python-linux-cuda"
        exit 1
    fi
else
    VENV=/opt/venv
    echo "[entrypoint] Using image venv: $VENV"
    if [ ! -f "$VENV/bin/python3" ]; then
        echo "[entrypoint] ERROR: /opt/venv not found"
        echo "[entrypoint]   Was the image built with DEV_VENV=0 (the default)?"
        echo "[entrypoint]   If using DEV_VENV=1, set it as a runtime env var too, not just a build arg."
        exit 1
    fi
fi

export PATH="$VENV/bin:$PATH"
echo "[entrypoint] python: $(which python3)"
echo "[entrypoint] exec: $*"
exec "$@"
