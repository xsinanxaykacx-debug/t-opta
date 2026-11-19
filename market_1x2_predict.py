"""Turn upcoming 1X2 odds into discovery-gated market-favorite picks and coupons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_1x2_analysis import (
    PROBABILITY_BAND_LABELS,
    market_probabilities,
)


def _band_for_probability(probability: float) -> str:
    if pd.isna(probability):
        return "no_valid_odds"
    if probability < 0.60:
        return "below_60%"
    for label, lower, upper in zip(
        PROBABILITY_BAND_LABELS, (0.60, 0.71, 0.81, 0.91), (0.71, 0.81, 0.91, 1.0000001)
    ):
        if lower <= probability < upper:
            return label
    return "above_100%"


def predict_upcoming(frame: pd.DataFrame, selected_bands: list[dict], output_dir: Path) -> dict:
    """Predict the market favorite from odds only; no result/feature columns required."""
    if frame.empty:
        raise ValueError("Upcoming fixture CSV is empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    market, source_info = market_probabilities(frame)
    data = pd.concat([frame.reset_index(drop=True), market.reset_index(drop=True)], axis=1)
    probability_matrix = data[["p_home", "p_draw", "p_away"]].to_numpy(float)
    valid_mask = data["odds_valid"].to_numpy(dtype=bool)
    favorite_codes = np.argmax(probability_matrix, axis=1)
    favorite_codes[~valid_mask] = -1
    favorite_names = np.array(["MS1", "MSX", "MS2"])
    data["market_favorite_code"] = favorite_codes
    data["market_favorite"] = np.where(
        valid_mask, favorite_names[np.maximum(favorite_codes, 0)], "NO_ODDS"
    )
    favorite_probability = np.full(len(data), np.nan, dtype=float)
    favorite_probability[valid_mask] = np.max(probability_matrix[valid_mask], axis=1)
    data["favorite_probability"] = favorite_probability
    data["favorite_probability_pct"] = data["favorite_probability"] * 100.0
    data["probability_band"] = data["favorite_probability"].map(_band_for_probability)
    selected_labels = {item["band"] for item in selected_bands}
    data["selected_for_coupon"] = data["probability_band"].isin(selected_labels)
    odds_matrix = data[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    favorite_odds = np.full(len(data), np.nan, dtype=float)
    favorite_odds[valid_mask] = odds_matrix[np.arange(len(data))[valid_mask], favorite_codes[valid_mask]]
    data["favorite_odds"] = favorite_odds
    data["selection_reason"] = np.where(
        data["selected_for_coupon"],
        "Band passed discovery sample/hit-rate gate",
        "Outside discovery-selected bands",
    )

    # Preserve every input fixture in the picks file; rank eligible picks by raw probability.
    ranked = data.loc[data["selected_for_coupon"]].sort_values(
        ["favorite_probability"], ascending=False, kind="stable"
    ).head(10).copy()
    ranked.insert(0, "pick_rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(output_dir / "upcoming_market_picks.csv", index=False)
    data.to_csv(output_dir / "all_upcoming_market_favorites.csv", index=False)

    coupon_rows = []
    coupon_summary = []
    if len(ranked) >= 10:
        # Interleave ranks so the three coupons receive similarly ranked selections.
        for coupon_index, coupon_name in enumerate(("Kupon 1", "Kupon 2", "Kupon 3")):
            indices = [coupon_index, coupon_index + 3, coupon_index + 6]
            legs = ranked.iloc[indices]
            combined_odds = float(legs["favorite_odds"].prod())
            for leg_number, (_, match) in enumerate(legs.iterrows(), start=1):
                fixture = match.get("Fixture", match.get("Match", None))
                if fixture is None or pd.isna(fixture) or not str(fixture).strip():
                    home = match.get("HomeTeam", None)
                    away = match.get("AwayTeam", None)
                    if home is not None and away is not None:
                        fixture = f"{home} - {away}"
                    elif home is not None:
                        fixture = home
                    else:
                        fixture = f"Pick {match['pick_rank']}"
                coupon_rows.append({
                    "coupon": coupon_name,
                    "leg": leg_number,
                    "pick_rank": int(match["pick_rank"]),
                    "fixture": str(fixture),
                    "market_favorite": match["market_favorite"],
                    "raw_implied_probability_pct": float(match["favorite_probability_pct"]),
                    "favorite_odds": float(match["favorite_odds"]),
                })
            coupon_summary.append({
                "coupon": coupon_name,
                "legs": 3,
                "combined_decimal_odds": combined_odds,
                "stake_per_coupon": 100.0,
                "potential_return": combined_odds * 100.0,
                "potential_net_if_won": combined_odds * 100.0 - 300.0,
            })
        unused = ranked.iloc[9]
        unused_pick = {
            "pick_rank": int(unused["pick_rank"]),
            "market_favorite": unused["market_favorite"],
            "raw_implied_probability_pct": float(unused["favorite_probability_pct"]),
            "favorite_odds": float(unused["favorite_odds"]),
        }
        status = "ready"
    else:
        unused_pick = None
        status = "fewer_than_10_eligible_fixtures"
    pd.DataFrame(coupon_rows, columns=[
        "coupon", "leg", "pick_rank", "fixture", "market_favorite",
        "raw_implied_probability_pct", "favorite_odds"
    ]).to_csv(output_dir / "upcoming_coupons.csv", index=False)
    pd.DataFrame(coupon_summary, columns=[
        "coupon", "legs", "combined_decimal_odds", "stake_per_coupon",
        "potential_return", "potential_net_if_won"
    ]).to_csv(output_dir / "upcoming_coupon_summary.csv", index=False)

    result = {
        "status": status,
        "fixtures_received": int(len(data)),
        "fixtures_with_complete_odds": int(data["odds_valid"].sum()),
        "eligible_fixtures": int(data["selected_for_coupon"].sum()),
        "picks_in_group": int(len(ranked)),
        "selected_bands_from_discovery": selected_bands,
        "selected_odds_source": source_info,
        "coupon_count": len(coupon_summary),
        "coupon_stake_total": 300.0 if coupon_summary else 0.0,
        "coupons": coupon_summary,
        "unused_tenth_pick": unused_pick,
        "method": "Market favorite = highest raw 1/decimal-odds value; no de-vigging or normalization.",
        "warning": "Only enter odds captured before kickoff. This script does not independently verify odds timestamps or fixture status.",
    }
    (output_dir / "upcoming_prediction_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Upcoming 1X2 market-favorite picker")
    parser.add_argument("csv", type=Path, help="Upcoming fixtures CSV with complete H/D/A odds triplets")
    parser.add_argument("--model-summary", type=Path, required=True,
                        help="market_1x2_summary.json generated by historical analysis")
    parser.add_argument("--out", type=Path, default=Path("results_upcoming"))
    args = parser.parse_args()
    model_summary = json.loads(args.model_summary.read_text(encoding="utf-8"))
    selected_bands = model_summary.get("selected_candidate_bands_from_discovery", [])
    frame = pd.read_csv(args.csv, encoding="latin-1", low_memory=False)
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(predict_upcoming(frame, selected_bands, args.out),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
