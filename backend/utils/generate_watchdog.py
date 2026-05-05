from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
import threading

from .. import config
from ..constants import ENV_SERVERLESS, GENERATE_WATCHDOG_FILENAME, GENERATE_WATCHDOG_WINDOW_SECONDS

logger = logging.getLogger(__name__)


class GenerateWatchdog:
    def __init__(self, state_path: Path, window_seconds: int):
        self.state_path = state_path
        self.window_seconds = window_seconds
        self._last_success_at: datetime | None = None
        self._lock = threading.Lock()

    def record_success(self, completed_at: datetime, generation_id: str) -> bool:
        return self._record_success(completed_at, generation_id, self.state_path)

    def _record_success(self, completed_at: datetime, generation_id: str, state_path: Path) -> bool:
        with self._lock:
            previous_success_at = self._last_success_at
            if previous_success_at is not None and completed_at < previous_success_at:
                return False

            should_write = (
                previous_success_at is not None
                and (completed_at - previous_success_at).total_seconds() <= self.window_seconds
            )
            self._last_success_at = completed_at

            if not should_write:
                return False

            self._write_state_file(
                state_path,
                {
                    "timestamp": completed_at.isoformat(),
                    "generation_id": generation_id,
                },
            )

        logger.info(f"Generate watchdog state updated: {state_path}")
        return True

    def _write_state_file(self, state_path: Path, payload: dict[str, str]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)

        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=state_path.parent, delete=False) as temp_file:
                temp_name = temp_file.name
                json.dump(payload, temp_file, sort_keys=True)
                temp_file.write("\n")
                temp_file.flush()
            os.replace(temp_name, state_path)
        except Exception:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
            raise


_generate_watchdog = GenerateWatchdog(
    state_path=Path(GENERATE_WATCHDOG_FILENAME),
    window_seconds=GENERATE_WATCHDOG_WINDOW_SECONDS,
)


def _is_serverless() -> bool:
    return os.environ.get(ENV_SERVERLESS, "0") == "1"


def record_generate_success(generation_id: str, completed_at: datetime | None = None) -> bool:
    if _is_serverless():
        return False

    return _generate_watchdog._record_success(
        completed_at or datetime.utcnow(),
        generation_id,
        config.get_data_dir() / GENERATE_WATCHDOG_FILENAME,
    )
