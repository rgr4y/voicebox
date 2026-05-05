from datetime import datetime
import json

import pytest

from backend.utils import generate_watchdog
from backend.utils.generate_watchdog import GenerateWatchdog


@pytest.fixture(autouse=True)
def clear_serverless_env(monkeypatch):
    monkeypatch.delenv(generate_watchdog.ENV_SERVERLESS, raising=False)


def test_first_success_does_not_write_state_file(tmp_path):
    state_path = tmp_path / ".voicebox.generate-watchdog"
    watchdog = GenerateWatchdog(state_path=state_path, window_seconds=600)

    wrote = watchdog.record_success(datetime(2026, 5, 5, 12, 0, 0), generation_id="gen-1")

    assert wrote is False
    assert not state_path.exists()


def test_second_success_within_window_writes_state_file(tmp_path):
    state_path = tmp_path / ".voicebox.generate-watchdog"
    watchdog = GenerateWatchdog(state_path=state_path, window_seconds=600)

    watchdog.record_success(datetime(2026, 5, 5, 12, 0, 0), generation_id="gen-1")
    wrote = watchdog.record_success(datetime(2026, 5, 5, 12, 9, 59), generation_id="gen-2")

    assert wrote is True
    payload = json.loads(state_path.read_text())
    assert payload["generation_id"] == "gen-2"
    assert payload["timestamp"] == "2026-05-05T12:09:59"


def test_second_success_at_window_boundary_writes_state_file(tmp_path):
    state_path = tmp_path / ".voicebox.generate-watchdog"
    watchdog = GenerateWatchdog(state_path=state_path, window_seconds=600)

    watchdog.record_success(datetime(2026, 5, 5, 12, 0, 0), generation_id="gen-1")
    wrote = watchdog.record_success(datetime(2026, 5, 5, 12, 10, 0), generation_id="gen-2")

    assert wrote is True
    payload = json.loads(state_path.read_text())
    assert payload["generation_id"] == "gen-2"
    assert payload["timestamp"] == "2026-05-05T12:10:00"


def test_second_success_after_window_does_not_write_state_file(tmp_path):
    state_path = tmp_path / ".voicebox.generate-watchdog"
    watchdog = GenerateWatchdog(state_path=state_path, window_seconds=600)

    watchdog.record_success(datetime(2026, 5, 5, 12, 0, 0), generation_id="gen-1")
    wrote = watchdog.record_success(datetime(2026, 5, 5, 12, 10, 1), generation_id="gen-2")

    assert wrote is False
    assert not state_path.exists()


def test_non_writing_success_resets_clock(tmp_path):
    state_path = tmp_path / ".voicebox.generate-watchdog"
    watchdog = GenerateWatchdog(state_path=state_path, window_seconds=600)

    watchdog.record_success(datetime(2026, 5, 5, 12, 0, 0), generation_id="gen-1")
    watchdog.record_success(datetime(2026, 5, 5, 12, 10, 1), generation_id="gen-2")
    wrote = watchdog.record_success(datetime(2026, 5, 5, 12, 20, 0), generation_id="gen-3")

    assert wrote is True
    payload = json.loads(state_path.read_text())
    assert payload["generation_id"] == "gen-3"
    assert payload["timestamp"] == "2026-05-05T12:20:00"


def test_older_success_does_not_move_clock_backward_or_write_file(tmp_path):
    state_path = tmp_path / ".voicebox.generate-watchdog"
    watchdog = GenerateWatchdog(state_path=state_path, window_seconds=600)

    watchdog.record_success(datetime(2026, 5, 5, 12, 10, 0), generation_id="gen-newer")
    wrote_older = watchdog.record_success(datetime(2026, 5, 5, 12, 9, 59), generation_id="gen-older")
    wrote_later = watchdog.record_success(datetime(2026, 5, 5, 12, 20, 0), generation_id="gen-later")

    assert wrote_older is False
    assert wrote_later is True
    payload = json.loads(state_path.read_text())
    assert payload["generation_id"] == "gen-later"
    assert payload["timestamp"] == "2026-05-05T12:20:00"


def test_record_generate_success_uses_current_data_dir(tmp_path, monkeypatch):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    bogus_path = tmp_path / "bogus" / generate_watchdog.GENERATE_WATCHDOG_FILENAME
    monkeypatch.setattr(
        generate_watchdog,
        "_generate_watchdog",
        GenerateWatchdog(state_path=bogus_path, window_seconds=600),
    )

    data_dirs = iter([first_dir, second_dir])
    monkeypatch.setattr(generate_watchdog.config, "get_data_dir", lambda: next(data_dirs))

    first_wrote = generate_watchdog.record_generate_success(
        generation_id="gen-1",
        completed_at=datetime(2026, 5, 5, 12, 0, 0),
    )
    second_wrote = generate_watchdog.record_generate_success(
        generation_id="gen-2",
        completed_at=datetime(2026, 5, 5, 12, 9, 59),
    )

    first_path = first_dir / generate_watchdog.GENERATE_WATCHDOG_FILENAME
    second_path = second_dir / generate_watchdog.GENERATE_WATCHDOG_FILENAME

    assert first_wrote is False
    assert second_wrote is True
    assert not bogus_path.exists()
    assert not first_path.exists()
    assert second_path.exists()
    payload = json.loads(second_path.read_text())
    assert payload["generation_id"] == "gen-2"
    assert payload["timestamp"] == "2026-05-05T12:09:59"


def test_module_record_generate_success_uses_singleton(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        generate_watchdog,
        "_generate_watchdog",
        GenerateWatchdog(state_path=tmp_path / "stale", window_seconds=600),
    )
    monkeypatch.setattr(generate_watchdog.config, "get_data_dir", lambda: data_dir)

    first_wrote = generate_watchdog.record_generate_success(
        "gen-4",
        completed_at=datetime(2026, 5, 5, 13, 0, 0),
    )
    second_wrote = generate_watchdog.record_generate_success(
        "gen-5",
        completed_at=datetime(2026, 5, 5, 13, 9, 0),
    )

    state_path = data_dir / generate_watchdog.GENERATE_WATCHDOG_FILENAME
    assert first_wrote is False
    assert second_wrote is True
    payload = json.loads(state_path.read_text())
    assert payload["generation_id"] == "gen-5"
    assert payload["timestamp"] == "2026-05-05T13:09:00"


def test_record_generate_success_is_noop_in_serverless(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv(generate_watchdog.ENV_SERVERLESS, "1")
    monkeypatch.setattr(
        generate_watchdog,
        "_generate_watchdog",
        GenerateWatchdog(state_path=tmp_path / "stale", window_seconds=600),
    )
    monkeypatch.setattr(generate_watchdog.config, "get_data_dir", lambda: data_dir)

    first_wrote = generate_watchdog.record_generate_success(
        "gen-serverless-1",
        completed_at=datetime(2026, 5, 5, 14, 0, 0),
    )
    second_wrote = generate_watchdog.record_generate_success(
        "gen-serverless-2",
        completed_at=datetime(2026, 5, 5, 14, 9, 0),
    )

    state_path = data_dir / generate_watchdog.GENERATE_WATCHDOG_FILENAME
    assert first_wrote is False
    assert second_wrote is False
    assert not state_path.exists()
