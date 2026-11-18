import numpy as np
import pandas as pd

from scripts.predict_upcoming import (
    build_prediction_table,
    prepare_upcoming_features,
)


def test_prediction_table_returns_only_top_confident_fixtures():
    upcoming = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-11-20T12:00:00Z",
            "2026-11-20T13:00:00Z",
            "2026-11-20T14:00:00Z",
            "2026-11-20T15:00:00Z",
        ], utc=True),
        "League": ["L1"] * 4,
        "HomeTeam": ["A", "B", "C", "D"],
        "AwayTeam": ["E", "F", "G", "H"],
        "lambda_home": [2.0, 0.5, 1.5, 1.2],
        "lambda_away": [1.0, 0.5, 1.0, 1.1],
        "p_poisson": [0.80, 0.20, 0.72, 0.58],
    })

    result = build_prediction_table(
        upcoming,
        confidence_cutoff=0.70,
        top_n=2,
    )

    picks = result.loc[result["status"] == "PICK"]
    assert len(picks) == 2
    assert set(picks["HomeTeam"]) == {"A", "B"}
    assert set(picks["predicted_side"]) == {"OVER", "UNDER"}
    assert (picks["confidence"] >= 0.70).all()


def test_upcoming_predictions_exclude_all_same_day_history():
    history = pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-12-31T12:00:00Z",
            "2025-01-01T09:00:00Z",
        ], utc=True),
        "League": ["L1", "L1"],
        "HomeTeam": ["X", "P"],
        "AwayTeam": ["Y", "Q"],
        "Home_xG": [4.0, 0.5],
        "Away_xG": [4.0, 0.5],
        "FTHG": [4, 0],
        "FTAG": [4, 0],
    })
    upcoming = pd.DataFrame({
        "Date": pd.to_datetime([
            "2025-01-01T15:00:00Z",
            "2025-01-01T18:00:00Z",
        ], utc=True),
        "League": ["L1", "L1"],
        "HomeTeam": ["A", "C"],
        "AwayTeam": ["B", "D"],
    })

    features = prepare_upcoming_features(history, upcoming)

    assert len(features) == 2
    assert np.allclose(features["league_average"].to_numpy(), [4.0, 4.0])
    assert np.allclose(features["lambda_home"].to_numpy(), [4.0, 4.0])
    assert np.allclose(features["p_poisson"].to_numpy(), features["p_poisson"].iloc[0])


def test_prediction_table_uses_supplied_recent_blend_probabilities():
    upcoming = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-11-20T12:00:00Z",
            "2026-11-20T13:00:00Z",
            "2026-11-20T14:00:00Z",
        ], utc=True),
        "League": ["L1"] * 3,
        "HomeTeam": ["A", "B", "C"],
        "AwayTeam": ["D", "E", "F"],
        "lambda_home": [1.0, 1.0, 1.0],
        "lambda_away": [1.0, 1.0, 1.0],
        "p_poisson": [0.95, 0.05, 0.50],
    })

    result = build_prediction_table(
        upcoming,
        confidence_cutoff=0.70,
        top_n=2,
        probabilities=np.array([0.55, 0.91, 0.12]),
    )
    picks = result.loc[result["status"] == "PICK"]

    assert set(picks["HomeTeam"]) == {"B", "C"}
    assert np.allclose(
        result.set_index("HomeTeam").loc[["A", "B", "C"], "p_over"].to_numpy(),
        [0.55, 0.91, 0.12],
    )


def test_predict_upcoming_calibrates_recent_blend_from_last_500_history(tmp_path):
    from scripts.predict_upcoming import predict_upcoming

    size = 510
    labels = np.arange(size) % 2
    history = pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=size, freq="D", tz="UTC"),
        "League": ["L1"] * size,
        "HomeTeam": [f"T{index % 20}" for index in range(size)],
        "AwayTeam": [f"T{(index % 20 + 1) % 20}" for index in range(size)],
        "Home_xG": np.where(labels == 1, 1.8, 0.8),
        "Away_xG": np.where(labels == 1, 1.0, 0.7),
        "FTHG": np.where(labels == 1, 2, 0),
        "FTAG": np.where(labels == 1, 1, 1),
    })
    upcoming = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01T18:00:00Z"], utc=True),
        "League": ["L1"],
        "HomeTeam": ["T1"],
        "AwayTeam": ["T2"],
    })
    history_path = tmp_path / "history.csv"
    upcoming_path = tmp_path / "upcoming.csv"
    output_path = tmp_path / "predictions.csv"
    history.to_csv(history_path, index=False)
    upcoming.to_csv(upcoming_path, index=False)

    result = predict_upcoming(
        history_path,
        upcoming_path,
        output_path=output_path,
        confidence_cutoff=0.70,
        top_n=10,
    )

    assert len(result) == 1
    assert output_path.exists()
    assert np.isfinite(result["p_over"]).all()
    assert np.isfinite(result["p_under"]).all()
