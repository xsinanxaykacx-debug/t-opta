from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from mackolik_provider import collect_upcoming
from market_1x2_analysis import market_probabilities
from market_1x2_predict import _band_for_probability, predict_upcoming
from modules.odds_similarity_engine import find_similar_matches

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
SUMMARY_PATH = ROOT / "results_market_1x2" / "market_1x2_summary.json"
HISTORICAL_TYPES_PATH = ROOT / "results_market_1x2" / "market_1x2_favorite_types.csv"
HISTORY_CSV_PATH = ROOT / "xgo.csv"
OUTPUT_DIR = ROOT / "results_upcoming"

app = Flask(__name__, static_folder=None)


def _load_historical_performance():
    if not HISTORICAL_TYPES_PATH.exists() or not SUMMARY_PATH.exists():
        raise RuntimeError(
            "Geçmiş kazanma/kaybetme analizi bulunamadı. Önce run_bot.bat ile tarihsel analizi çalıştırın."
        )
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(HISTORICAL_TYPES_PATH)
    stats = {}
    for row in frame.to_dict(orient="records"):
        count = int(row["matches"])
        wins = int(row["wins"])
        rate = row.get("hit_rate")
        lower = row.get("hit_rate_ci95_lower")
        upper = row.get("hit_rate_ci95_upper")
        partition = str(row["partition"])
        record = {
            "sample_size": count,
            "wins": wins,
            "losses": count - wins,
            "hit_rate_pct": round(float(rate) * 100, 2) if pd.notna(rate) else None,
            "ci95_lower_pct": round(float(lower) * 100, 2) if pd.notna(lower) else None,
            "ci95_upper_pct": round(float(upper) * 100, 2) if pd.notna(upper) else None,
        }
        stats.setdefault(
            (str(row["favorite_probability_band"]), str(row["favorite_type"])), {}
        )[partition] = record
    validation_meta = summary.get("validation", {})
    return stats, {
        "matches": int(validation_meta.get("rows", 0)),
        "start": validation_meta.get("start"),
        "end": validation_meta.get("end"),
        "odds_source": summary.get("selected_odds_source", {}).get("source_counts", {}),
        "note": "Geçmiş oran arşivi ile Maçkolik canlı oranları farklı kaynaklardır; oran kaynağı aktarımı ayrıca izlenmelidir.",
    }


def _load_selected_bands():
    if not SUMMARY_PATH.exists():
        raise RuntimeError(
            "Geçmiş piyasa analizi bulunamadı. Önce run_bot.bat ile ilk kurulumu tamamlayın."
        )
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return summary.get("selected_candidate_bands_from_discovery", [])


def _analyze_rows(rows: list[dict]) -> dict:
    selected_bands = _load_selected_bands()
    historical_stats, historical_meta = _load_historical_performance()
    if not rows:
        return {
            "status": "success",
            "date": pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d"),
            "fixtures_found": 0,
            "fixtures_with_odds": 0,
            "eligible_picks": 0,
            "coupon_ready": False,
            "message": "Maçkolik'te bugün oranı alınabilen maç bulunamadı.",
            "historical_validation": historical_meta,
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
    for _, row in analyzed.iterrows():
        probabilities = {
            "1": float(row["p_home"]),
            "X": float(row["p_draw"]),
            "2": float(row["p_away"]),
        }
        favorite_key = max(probabilities, key=probabilities.get)
        favorite_type = {"1": "MS1", "X": "MSX", "2": "MS2"}[favorite_key]
        favorite_probability = probabilities[favorite_key]
        band = _band_for_probability(favorite_probability)
        historical_performance = historical_stats.get((band, favorite_type), {})
        historical_performance = {
            "validation": historical_performance.get("validation", {
                "sample_size": 0, "wins": 0, "losses": 0,
                "hit_rate_pct": None, "ci95_lower_pct": None, "ci95_upper_pct": None,
            }),
            "discovery": historical_performance.get("discovery", {
                "sample_size": 0, "wins": 0, "losses": 0, "hit_rate_pct": None,
            }),
        }
        historical_performance.update(historical_performance["validation"])
        historical_performance["partition"] = "chronological_validation"
        is_eligible = (band, favorite_type) in selected_pairs
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
            "favorite": favorite_type,
            "favorite_probability_pct": round(favorite_probability * 100, 2),
            "probability_band": band,
            "historical_performance": historical_performance,
            "eligible": is_eligible,
            "selection_note": "Tarihsel keşif kriterlerini geçti" if is_eligible else "Tarihsel seçim kriterlerini geçmedi",
            "match_url": str(row.get("MatchURL", "")),
            "odds_captured_at": str(row.get("OddsCapturedAt", "")),
        })

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
        "historical_validation": historical_meta,
        "probability_method": "Marjdan arındırılmış 1X2 olasılığı: her 1/oran değeri üç sonucun 1/oran toplamına bölünür.",
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


@app.get("/similar.js")
def similar_script():
    return send_from_directory(WEB_DIR, "similar.js")


@app.get("/api/fixtures/today")
def fixtures_today():
    try:
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


@app.post("/api/matches/similar")
def similar_matches():
    try:
        payload = request.get_json(silent=True) or {}
        raw_odds = payload.get("odds") or {}\n        odds = {\n            "home": raw_odds.get("1", raw_odds.get("home")),\n            "draw": raw_odds.get("X", raw_odds.get("draw")),\n            "away": raw_odds.get("2", raw_odds.get("away")),\n        }
        target_date = payload.get("date")
        limit = min(max(int(payload.get("limit", 20)), 1), 50)
        if not target_date:
            return jsonify({"status": "error", "message": "Maç tarihi eksik."}), 400
        if not HISTORY_CSV_PATH.is_file():
            return jsonify({
                "status": "error",
                "message": "xgo.csv bulunamadı. Geçmiş oran arşivini T-OPTA ana klasörüne koyun.",
            }), 503

        history = pd.read_csv(HISTORY_CSV_PATH, encoding="latin-1", low_memory=False)
        similar = find_similar_matches(history, odds, target_date, limit=limit)
        results = []
        for row in similar.to_dict(orient="records"):
            match_date = pd.to_datetime(row.get("Date"), utc=True, errors="coerce")
            results.append({
                "date": match_date.strftime("%Y-%m-%d") if pd.notna(match_date) else "",
                "league": str(row.get("League", "")),
                "home": str(row.get("HomeTeam", "")),
                "away": str(row.get("AwayTeam", "")),
                "score": f'{int(row["FTHG"])}-{int(row["FTAG"])}',
                "result": str(row.get("actual_result", "")),
                "odds": {
                    "1": round(float(row["odds_home"]), 2),
                    "X": round(float(row["odds_draw"]), 2),
                    "2": round(float(row["odds_away"]), 2),
                },
                "distance_pp": round(float(row["probability_distance_pp"]), 3),
            })
        return jsonify({
            "status": "success",
            "count": len(results),
            "matches": results,
            "note": "Benzerlik, marjdan arındırılmış 1X2 olasılıklarına göre hesaplanır. Arşivde oran yakalama zaman damgası bulunmadığından oranların maç öncesi alındığı bağımsız doğrulanamaz.",
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Similar-match lookup failed")
        return jsonify({"status": "error", "message": f"Benzer maç analizi yapılamadı: {exc}"}), 500


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
