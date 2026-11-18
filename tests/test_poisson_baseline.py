import pandas as pd

from experiments.poisson_baseline import PoissonBaseline


def _same_day_fixture(first_home_xg: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2024-01-01T12:00:00Z",
                "League": "L",
                "HomeTeam": "A",
                "AwayTeam": "B",
                "Home_xG": 2.0,
                "Away_xG": 1.0,
            },
            {
                "Date": "2024-01-01T15:00:00Z",
                "League": "L",
                "HomeTeam": "C",
                "AwayTeam": "D",
                "Home_xG": 1.5,
                "Away_xG": 1.5,
            },
            {
                "Date": "2024-01-01T18:00:00Z",
                "League": "L",
                "HomeTeam": "E",
                "AwayTeam": "F",
                "Home_xG": 0.5,
                "Away_xG": 0.5,
            },
            {
                "Date": "2024-01-02T12:00:00Z",
                "League": "L",
                "HomeTeam": "A",
                "AwayTeam": "C",
                "Home_xG": first_home_xg,
                "Away_xG": 0.1,
            },
            {
                "Date": "2024-01-02T12:00:00Z",
                "League": "L",
                "HomeTeam": "B",
                "AwayTeam": "D",
                "Home_xG": 0.2,
                "Away_xG": 4.0,
            },
        ]
    )


def test_same_calendar_day_results_do_not_change_same_day_poisson_features():
    low = PoissonBaseline.add_historical_features(_same_day_fixture(0.1))
    high = PoissonBaseline.add_historical_features(_same_day_fixture(5.0))

    # The second 12:00 fixture must not see the first fixture's same-day xG.
    assert low.loc[4, "p_poisson"] == high.loc[4, "p_poisson"]


def test_poisson_probabilities_are_finite_and_bounded():
    featured = PoissonBaseline.add_historical_features(_same_day_fixture(1.2))
    probabilities = PoissonBaseline().predict_proba(featured)
    assert len(probabilities) == len(featured)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
