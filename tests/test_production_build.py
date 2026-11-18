"""Production artifact must remain pinned to the validated experiment evidence."""

import json

import pytest

from experiments import train_production


def _write_evidence(tmp_path, monkeypatch, *, commit="32abf6b", run_id=44330945792):
    summary_path = tmp_path / "experiment_summary.json"
    roi_path = tmp_path / "final_test_roi.json"

    summary = {
        "experiment_run_id": run_id,
        "experiment_commit": commit,
        "ready": True,
        "selected_model_on_policy_select": "odds_aware",
        "policy_selection": {
            "odds_aware": {"threshold": 0.735, "bets": 12, "hit_rate": 0.75}
        },
        "policy_check": {"odds_aware": {"target_met": True}},
        "final_test": {
            "odds_aware": {
                "selective": {
                    "target_met": True,
                    "bets": 450,
                    "hit_rate": 0.66,
                }
            }
        },
    }
    roi = {
        "selected_model": "odds_aware",
        "threshold": 0.735,
        "bets": 450,
        "hit_rate": 0.66,
        "roi_per_unit": -0.058977777777777775,
        "average_odds": 1.4494888888888888,
        "net_units": -26.54,
    }

    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    roi_path.write_text(json.dumps(roi), encoding="utf-8")
    monkeypatch.setattr(train_production, "EVIDENCE_PATH", summary_path)
    monkeypatch.setattr(train_production, "ROI_PATH", roi_path)


def test_production_build_accepts_only_the_frozen_experiment_evidence(
    tmp_path, monkeypatch
):
    _write_evidence(tmp_path, monkeypatch)

    summary, roi = train_production._load_and_verify_evidence()

    assert summary["experiment_run_id"] == 44330945792
    assert summary["experiment_commit"] == "32abf6b"
    assert roi["roi_per_unit"] < 0


@pytest.mark.parametrize(
    ("commit", "run_id"),
    [
        ("different", 44330945792),
        ("32abf6b", 1),
    ],
)
def test_production_build_rejects_unvalidated_evidence(
    tmp_path, monkeypatch, commit, run_id
):
    _write_evidence(
        tmp_path,
        monkeypatch,
        commit=commit,
        run_id=run_id,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        train_production._load_and_verify_evidence()
