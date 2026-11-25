from datetime import datetime, timezone

from modules.odds_snapshot_store import build_match_key, get_snapshots, record_snapshot


def test_snapshot_store_records_only_odds_changes(tmp_path):
    db_path = tmp_path / "odds.sqlite3"
    key = build_match_key("fixture-1", "", "2026-11-25", "League", "Home", "Away")
    first_time = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
    second_time = datetime(2026, 11, 25, 10, 5, tzinfo=timezone.utc)

    assert record_snapshot(key, {"1": 1.8, "X": 3.4, "2": 4.2}, db_path, captured_at=first_time)
    assert not record_snapshot(key, {"1": 1.8, "X": 3.4, "2": 4.2}, db_path, captured_at=second_time)
    assert record_snapshot(key, {"1": 1.9, "X": 3.3, "2": 4.1}, db_path, captured_at=second_time)

    history = get_snapshots(key, db_path)
    assert len(history) == 2
    assert history[0]["odds"]["1"] == 1.8
    assert history[1]["odds"]["1"] == 1.9


def test_match_key_falls_back_to_fixture_identity():
    first = build_match_key("", "", "2026-11-25 20:00", "League", "Home", "Away")
    second = build_match_key("", "", "2026-11-25 20:00", "League", "Home", "Away")
    assert first == second


def test_snapshot_store_rejects_invalid_odds(tmp_path):
    try:
        record_snapshot("fixture", {"1": 1.0, "X": 3.0, "2": 4.0}, tmp_path / "bad.sqlite3")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid decimal odds must be rejected.")
