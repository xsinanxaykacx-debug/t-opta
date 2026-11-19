import json

import pandas as pd

import mackolik_web


def test_dashboard_history_loader_uses_validation_wins_and_losses(tmp_path, monkeypatch):
    types_path = tmp_path / "market_1x2_favorite_types.csv"
    summary_path = tmp_path / "market_1x2_summary.json"
    pd.DataFrame([
        {
            "partition": "discovery", "favorite_probability_band": "81.00%-90.99%",
            "favorite_type": "MS1", "matches": 300, "wins": 250,
            "hit_rate": 250 / 300, "hit_rate_ci95_lower": 0.79, "hit_rate_ci95_upper": 0.87,
        },
        {
            "partition": "validation", "favorite_probability_band": "81.00%-90.99%",
            "favorite_type": "MS1", "matches": 61, "wins": 56,
            "hit_rate": 56 / 61, "hit_rate_ci95_lower": 0.82, "hit_rate_ci95_upper": 0.96,
        },
    ]).to_csv(types_path, index=False)
    summary_path.write_text(json.dumps({
        "validation": {"rows": 2112, "start": "2025-03-18", "end": "2025-11-01"},
        "selected_odds_source": {"source_counts": {"Avg": 14169}},
    }), encoding="utf-8")

    monkeypatch.setattr(mackolik_web, "HISTORICAL_TYPES_PATH", types_path)
    monkeypatch.setattr(mackolik_web, "SUMMARY_PATH", summary_path)

    stats, meta = mackolik_web._load_historical_performance()

    record = stats[("81.00%-90.99%", "MS1")]
    assert record["sample_size"] == 61
    assert record["wins"] == 56
    assert record["losses"] == 5
    assert record["hit_rate_pct"] == 91.8
    assert record["partition"] == "chronological_validation"
    assert record["validation"]["wins"] == 56
    assert record["discovery"]["sample_size"] == 300
    assert meta["matches"] == 2112
    assert meta["odds_source"] == {"Avg": 14169}
