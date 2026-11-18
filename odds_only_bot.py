"""
ODDS-ONLY high-odds Over/Under 2.5 model.

Only pre-match market odds are predictors. Match results are labels only.
The historical CSV does not contain reliable odds capture timestamps; the
program therefore warns that pre-kickoff timing must be verified separately.

Usage:
    python odds_only_bot.py xgo.csv --out results_odds_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_ODDS = 1.60
MAX_ODDS = 2.50
EV_THRESHOLDS = (0.00, 0.02, 0.05)
RANDOM_SEED = 42


def _valid_odds(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values[~np.isfinite(values) | (values <= 1.0)] = np.nan
    return values


def market_data(frame: pd.DataFrame):
    """Use one complete odds pair: Avg first, otherwise B365."""
    n = len(frame)

    def column(name):
        if name in frame.columns:
            return _valid_odds(frame[name])
        return np.full(n, np.nan, dtype=float)

    avg_over = column("odds_Avg>2.5")
    avg_under = column("odds_Avg<2.5")
    b365_over = column("odds_B365>2.5")
    b365_under = column("odds_B365<2.5")

    avg_ok = np.isfinite(avg_over) & np.isfinite(avg_under)
    b365_ok = np.isfinite(b365_over) & np.isfinite(b365_under)

    over = np.where(avg_ok, avg_over, np.where(b365_ok, b365_over, np.nan))
    under = np.where(avg_ok, avg_under, np.where(b365_ok, b365_under, np.nan))
    valid = np.isfinite(over) & np.isfinite(under)

    p_over = np.full(n, np.nan, dtype=float)
    inv_over = 1.0 / over[valid]
    inv_under = 1.0 / under[valid]
    p_over[valid] = inv_over / (inv_over + inv_under)

    margin = np.full(n, np.nan, dtype=float)
    margin[valid] = inv_over + inv_under - 1.0
    return p_over, over, under, margin, valid


def odds_features(frame: pd.DataFrame) -> np.ndarray:
    """Every predictor is calculated exclusively from the pre-match odds pair."""
    p_over, over, under, margin, valid = market_data(frame)
    features = np.column_stack([
        p_over,
        margin,
        np.log(over),
        np.log(under),
    ])
    features[~valid] = np.nan
    return features


def target(frame: pd.DataFrame) -> np.ndarray:
    goals = (
        pd.to_numeric(frame["FTHG"], errors="coerce").to_numpy(dtype=float)
        + pd.to_numeric(frame["FTAG"], errors="coerce").to_numpy(dtype=float)
    )
    return (goals > 2.5).astype(int)


def chronological_split(frame: pd.DataFrame, train_fraction=0.70, validation_fraction=0.15):
    """Split on complete UTC dates; the final 15% is deliberately left sealed."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split oranları 0 ile 1 arasında olmalı.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation toplamı 1'den küçük olmalı.")

    day = frame["Date"].dt.floor("D")
    counts = day.value_counts(sort=False).sort_index()
    if len(counts) < 3:
        raise ValueError("Kronolojik bölme için en az üç farklı UTC günü gerekli.")

    cumulative = counts.cumsum().to_numpy(dtype=int)
    train_cut = int(np.argmin(np.abs(cumulative - len(frame) * train_fraction)))
    val_target = len(frame) * (train_fraction + validation_fraction)
    val_candidates = np.arange(train_cut + 1, len(counts) - 1)
    if len(val_candidates) == 0:
        raise ValueError("Train/validation/holdout için yeterli gün yok.")
    val_cut = int(val_candidates[np.argmin(np.abs(cumulative[val_candidates] - val_target))])

    days = counts.index
    train_end, val_end = days[train_cut], days[val_cut]
    train = frame.loc[day <= train_end].copy()
    validation = frame.loc[(day > train_end) & (day <= val_end)].copy()
    holdout = frame.loc[day > val_end].copy()

    if min(len(train), len(validation), len(holdout)) == 0:
        raise ValueError("Bölümlerden biri boş kaldı.")
    assert train["Date"].dt.floor("D").max() < validation["Date"].dt.floor("D").min()
    assert validation["Date"].dt.floor("D").max() < holdout["Date"].dt.floor("D").min()
    return train, validation, holdout


def make_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
    ])


def _side_columns(frame, p_model):
    _, over, under, _, valid = market_data(frame)
    side = np.where(p_model >= 0.5, "OVER", "UNDER")
    selected_odds = np.where(side == "OVER", over, under)
    actual = target(frame)
    won = np.where(side == "OVER", actual == 1, actual == 0)
    eligible = valid & (selected_odds >= MIN_ODDS) & (selected_odds <= MAX_ODDS)

    over_ev = p_model * over - 1.0
    under_ev = (1.0 - p_model) * under - 1.0
    choose_over = over_ev >= under_ev
    ev_side = np.where(choose_over, "OVER", "UNDER")
    ev_odds = np.where(choose_over, over, under)
    ev_value = np.where(choose_over, over_ev, under_ev)
    ev_won = np.where(ev_side == "OVER", actual == 1, actual == 0)
    ev_eligible = valid & (ev_odds >= MIN_ODDS) & (ev_odds <= MAX_ODDS)

    return {
        "actual_over": actual,
        "side": side,
        "selected_odds": selected_odds,
        "won": won,
        "eligible": eligible,
        "over_ev": over_ev,
        "under_ev": under_ev,
        "ev_side": ev_side,
        "ev_odds": ev_odds,
        "ev_value": ev_value,
        "ev_won": ev_won,
        "ev_eligible": ev_eligible,
    }


def _roi_summary(odds: np.ndarray, won: np.ndarray):
    if len(odds) == 0:
        return {"bets": 0, "wins": 0, "hit_rate": np.nan, "roi_pct": np.nan, "profit_units": 0.0}
    profit = np.where(won, odds - 1.0, -1.0)
    return {
        "bets": int(len(odds)),
        "wins": int(np.sum(won)),
        "hit_rate": float(np.mean(won)),
        "roi_pct": float(np.sum(profit) / len(profit) * 100.0),
        "profit_units": float(np.sum(profit)),
    }


def evaluate_validation(frame: pd.DataFrame, p_model: np.ndarray, output_dir: Path):
    p_over, over, under, margin, valid = market_data(frame)
    y = target(frame)
    mask = valid & np.isfinite(p_model)

    if not mask.any():
        raise ValueError("Validation bölümünde geçerli oran çifti bulunamadı.")

    print("\n=== ODDS-ONLY VALIDATION ===")
    print(f"Geçerli oranlı maç: {int(mask.sum())}/{len(frame)}")
    print(f"Model accuracy (OVER/UNDER yönü): {accuracy_score(y[mask], p_model[mask] >= 0.5):.2%}")
    print(f"Model log-loss: {log_loss(y[mask], np.clip(p_model[mask], 1e-6, 1 - 1e-6), labels=[0, 1]):.5f}")
    print(f"Model Brier: {brier_score_loss(y[mask], p_model[mask]):.5f}")
    print(f"Market accuracy: {accuracy_score(y[mask], p_over[mask] >= 0.5):.2%}")
    print(f"Market log-loss: {log_loss(y[mask], np.clip(p_over[mask], 1e-6, 1 - 1e-6), labels=[0, 1]):.5f}")
    if np.unique(y[mask]).size == 2:
        print(f"Model AUC: {roc_auc_score(y[mask], p_model[mask]):.5f}")

    sides = _side_columns(frame, p_model)
    high = sides["eligible"] & mask
    high_result = _roi_summary(sides["selected_odds"][high], sides["won"][high])
    print(
        f"\n1.60–2.50 yön tahmini: n={high_result['bets']}, "
        f"isabet={high_result['hit_rate']:.2%}, ROI={high_result['roi_pct']:.2f}%"
        if high_result["bets"] else "\n1.60–2.50 aralığında yön tahmini yok."
    )

    bucket_rows = []
    bucket_edges = [1.60, 1.80, 2.00, 2.25, 2.50]
    for lower, upper in zip(bucket_edges[:-1], bucket_edges[1:]):
        in_bucket = high & (sides["selected_odds"] >= lower) & (
            sides["selected_odds"] < upper if upper < MAX_ODDS else sides["selected_odds"] <= upper
        )
        bucket = _roi_summary(sides["selected_odds"][in_bucket], sides["won"][in_bucket])
        bucket_rows.append({
            "odds_band": f"{lower:.2f}-{upper:.2f}",
            **bucket,
            "average_odds": float(np.mean(sides["selected_odds"][in_bucket])) if bucket["bets"] else np.nan,
        })
        if bucket["bets"]:
            print(
                f"Oran {lower:.2f}-{upper:.2f}: n={bucket['bets']}, "
                f"isabet={bucket['hit_rate']:.2%}, "
                f"ort.oran={bucket_rows[-1]['average_odds']:.2f}, "
                f"ROI={bucket['roi_pct']:.2f}%"
            )
        else:
            print(f"Oran {lower:.2f}-{upper:.2f}: maç yok")

    rows = []
    for threshold in EV_THRESHOLDS:
        selected = sides["ev_eligible"] & mask & (sides["ev_value"] >= threshold)
        result = _roi_summary(sides["ev_odds"][selected], sides["ev_won"][selected])
        rows.append({"min_expected_return": threshold, **result})
        if result["bets"]:
            print(
                f"EV >= {threshold:.0%}: n={result['bets']}, "
                f"isabet={result['hit_rate']:.2%}, ROI={result['roi_pct']:.2f}%, "
                f"net={result['profit_units']:+.2f} birim"
            )
        else:
            print(f"EV >= {threshold:.0%}: bahis yok")

    predictions = frame.loc[mask, [c for c in ["Date", "League", "HomeTeam", "AwayTeam"] if c in frame.columns]].copy()
    predictions["actual_over25"] = y[mask]
    predictions["p_market_over"] = p_over[mask]
    predictions["p_model_over"] = p_model[mask]
    predictions["market_margin"] = margin[mask]
    predictions["over_odds"] = over[mask]
    predictions["under_odds"] = under[mask]
    predictions["predicted_side"] = sides["side"][mask]
    predictions["selected_side_odds"] = sides["selected_odds"][mask]
    predictions["selected_side_won"] = sides["won"][mask]
    predictions["high_odds_eligible"] = sides["eligible"][mask]
    predictions["ev_best_side"] = sides["ev_side"][mask]
    predictions["ev_best_side_odds"] = sides["ev_odds"][mask]
    predictions["expected_return"] = sides["ev_value"][mask]
    predictions["ev_best_side_won"] = sides["ev_won"][mask]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "validation_ev_summary.csv", index=False)
    pd.DataFrame([{"strategy": "directional_1.60_to_2.50", **high_result}]).to_csv(
        output_dir / "validation_high_odds_summary.csv", index=False
    )
    pd.DataFrame(bucket_rows).to_csv(output_dir / "validation_odds_bands.csv", index=False)


def load_data(path: Path):
    frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
    required = ["Date", "FTHG", "FTAG"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("Eksik zorunlu sütunlar: " + ", ".join(missing))

    frame["Date"] = pd.to_datetime(frame["Date"], utc=True, errors="coerce")
    for name in ["FTHG", "FTAG"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    frame = frame.dropna(subset=["Date", "FTHG", "FTAG"]).copy()
    p_over, _, _, _, valid = market_data(frame)
    frame = frame.loc[valid & np.isfinite(p_over)].copy()
    frame = frame.sort_values(["Date"], kind="mergesort").reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError(f"Yeterli oranlı maç yok: {len(frame)}")
    return frame


def run(path: Path, output_dir: Path):
    frame = load_data(path)
    train, validation, sealed_holdout = chronological_split(frame)

    x_train = odds_features(train)
    y_train = target(train)
    valid_train = np.isfinite(x_train).all(axis=1)
    x_train = x_train[valid_train]
    y_train = y_train[valid_train]

    if np.unique(y_train).size != 2:
        raise ValueError("Train bölümünde OVER ve UNDER sonuçlarının ikisi de bulunmalı.")

    model = make_model()
    model.fit(x_train, y_train)

    x_validation = odds_features(validation)
    valid_validation = np.isfinite(x_validation).all(axis=1)
    if not valid_validation.any():
        raise ValueError("Validation bölümünde geçerli oran bulunamadı.")
    p_validation = np.full(len(validation), np.nan)
    p_validation[valid_validation] = model.predict_proba(x_validation[valid_validation])[:, 1]

    print(f"Toplam geçerli oranlı maç: {len(frame):,}")
    print(f"Train: {len(train):,} | Validation: {len(validation):,} | Kapalı holdout: {len(sealed_holdout):,}")
    print(f"Train tarihleri: {train.Date.min()} — {train.Date.max()}")
    print(f"Validation tarihleri: {validation.Date.min()} — {validation.Date.max()}")
    print(f"Holdout tarihleri (SONUÇLARI OKUNMADI): {sealed_holdout.Date.min()} — {sealed_holdout.Date.max()}")
    print("\nUYARI: CSV'de oran yakalama zaman damgası yok. Kullanılan Avg/B365 oranlarının maç başlamadan önce alındığı ayrıca doğrulanmalı.")
    evaluate_validation(validation, p_validation, output_dir)
    print(f"\nRaporlar yazıldı: {output_dir.resolve()}")
    print("Final holdout bu çalıştırmada bilerek değerlendirilmedi.")


def main():
    parser = argparse.ArgumentParser(description="Yalnızca maç önü oranlarıyla Over/Under 2.5 modeli")
    parser.add_argument("csv", type=Path, help="xgo.csv yolu")
    parser.add_argument("--out", type=Path, default=Path("results_odds_only"))
    args = parser.parse_args()
    run(args.csv, args.out)


if __name__ == "__main__":
    main()
