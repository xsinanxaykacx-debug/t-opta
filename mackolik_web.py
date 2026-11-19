from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, send_from_directory

from mackolik_provider import collect_upcoming
from market_1x2_analysis import market_probabilities
from market_1x2_predict import _band_for_probability, predict_upcoming

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
SUMMARY_PATH = ROOT / "results_market_1x2" / "market_1x2_summary.json"
OUTPUT_DIR = ROOT / "results_upcoming"

app = Flask(__name__, static_folder=None)


def _normalize_1x2_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Remove the 1X2 overround by scaling inverse odds to a 100% total."""
    total = sum(float(value) for value in probabilities.values())
    if total <= 0 or not all(float(value) > 0 for value in probabilities.values()):
        raise ValueError("1X2 olasılıkları pozitif ve toplamı sıfırdan büyük olmalı.")
    return {key: float(value) / total for key, value in probabilities.items()}


def _load_selected_bands():
    if not SUMMARY_PATH.exists():
        raise RuntimeError(
            "Geçmiş piyasa analizi bulunamadı. Önce run_bot.bat ile ilk kurulumu tamamlayın."
        )
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return summary.get("selected_candidate_bands_from_discovery", [])


def _analyze_rows(rows: list[dict]) -> dict:
    selected_bands = _load_selected_bands()
    if not rows:
        return {
            "status": "success",
            "date": pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d"),
            "fixtures_found": 0,
            "fixtures_with_odds": 0,
            "eligible_picks": 0,
            "coupon_ready": False,
            "message": "Maçkolik'te bugün oranı alınabilen maç bulunamadı.",
            "leagues": [],
            "matches": [],
        }

    frame = pd.DataFrame(rows)
    market, source_info = market_probabilities(frame)
    analyzed = pd.concat([frame.reset_index(drop=True), market.reset_index(drop=True)], axis=1)

    selected_pairs = {
        (item.get("band"), item.get("favorite_type"))
        for item in selected_bands
        if item.get("favorite_type") in {"MS1", "MSX", "MS2"}
    }

    matches = []
    for index, row in analyzed.iterrows():
        raw_probabilities = {
            "1": float(row["p_home"]),
            "X": float(row["p_draw"]),
            "2": float(row["p_away"]),
        }
        probabilities = _normalize_1x2_probabilities(raw_probabilities)
        favorite_key = max(probabilities, key=probabilities.get)
        favorite_type = {"1": "MS1", "X": "MSX", "2": "MS2"}[favorite_key]
        favorite_probability = probabilities[favorite_key]
        # Historical eligibility was calibrated on raw implied-probability bands;
        # keep it informational and independent from the displayed normalized bands.
        raw_favorite_probability = raw_probabilities[favorite_key]
        historical_band = _band_for_probability(raw_favorite_probability)
        band = _band_for_probability(favorite_probability)
        is_eligible = (historical_band, favorite_type) in selected_pairs
        matches.append({
            "match_id": str(row.get("MatchId", "")),
            "league": str(row.get("League", "Diğer ligler")),
            "date": str(row.get("Date", "")),
            "time": str(row.get("Date", "")).split(" ")[-1],
            "home": str(row.get("HomeTeam", "")),
            "away": str(row.get("AwayTeam", "")),
            "odds": {
                "1": float(row["odds_home"]),
                "X": float(row["odds_draw"]),
                "2": float(row["odds_away"]),
            },
            "probabilities_pct": {key: round(value * 100, 2) for key, value in probabilities.items()},
            "raw_probabilities_pct": {key: round(value * 100, 2) for key, value in raw_probabilities.items()},
            "favorite": favorite_type,
            "favorite_probability_pct": round(favorite_probability * 100, 2),
            "probability_band": band,
            "eligible": is_eligible,
            "selection_note": "Tarihsel keşif kriterlerini geçti" if is_eligible else "Tarihsel seçim kriterlerini geçmedi",
            "match_url": str(row.get("MatchURL", "")),
            "odds_captured_at": str(row.get("OddsCapturedAt", "")),
        })

    # Keep the same production prediction/coupon logic and write its CSV/JSON outputs.
    prediction_summary = predict_upcoming(frame, selected_bands, OUTPUT_DIR)
    eligible = [match for match in matches if match["eligible"]]
    matches.sort(key=lambda item: (item["date"], item["league"], item["home"]))

    league_map: dict[str, list[dict]] = {}
    for match in matches:
        league_map.setdefault(match["league"], []).append(match)

    count = len(eligible)
    if count >= 10:
        message = f"{count} uygun seçim bulundu. Aynı gün içinden 10 maçlık grup hazır."
    elif count > 0:
        message = f"10 adet uygun maç yok; {count} adet uygun seçim bulundu. Bulunanların tamamı listelendi."
    else:
        message = "Maçkolik'ten maçlar alındı ancak tarihsel seçim kriterlerini geçen maç yok."

    return {
        "status": "success",
        "date": pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d"),
        "fixtures_found": len(rows),
        "fixtures_with_odds": len(rows),
        "eligible_picks": count,
        "coupon_ready": prediction_summary.get("status") == "ready",
        "coupon_status": prediction_summary.get("status"),
        "message": message,
        "source": "Maçkolik",
        "source_info": source_info,
        "leagues": [
            {"league_name": league, "matches": league_matches}
            for league, league_matches in sorted(league_map.items())
        ],
        "matches": matches,
        "prediction_summary": prediction_summary,
    }


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/fixtures/today")
def fixtures_today():
    try:
        # Fetch the current day's fixtures only; do not substitute fabricated fallback data.
        rows = collect_upcoming(days=1)
        return jsonify(_analyze_rows(rows))
    except Exception as exc:
        return jsonify({
            "status": "error",
            "source": "Maçkolik",
            "message": f"Maçkolik verisi alınamadı: {exc}",
            "fixtures_found": 0,
            "eligible_picks": 0,
            "leagues": [],
            "matches": [],
        }), 502


@app.get("/outputs/<filename>")
def download_output(filename):
    allowed = {
        "upcoming_market_picks.csv",
        "all_upcoming_market_favorites.csv",
        "upcoming_coupons.csv",
        "upcoming_coupon_summary.csv",
        "upcoming_prediction_summary.json",
    }
    if filename not in allowed:
        abort(404)
    path = OUTPUT_DIR / filename
    if not path.is_file():
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "provider": "Mackolik"})


if __name__ == "__main__":
    print("T-OPTA web arayüzü: http://127.0.0.1:8765")
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
