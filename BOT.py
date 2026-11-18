
"""
bot.py — Leakage-safe Alt/Üst 2.5 tahmin ve value bet motoru.

Kullanım:
    python bot.py xgo.csv

Özellikler:
- Aynı maçın gerçekleşmiş xG değerleri tahmin girdisi değildir.
- Poisson takım güçleri yalnızca geçmiş maçlardan hesaplanır.
- Aynı gün oynanan maçlar birbirlerinin geçmişine eklenmez.
- Kalibrasyon / eşik seçimi / eşik doğrulaması ayrıdır.
- Final test yalnızca bir kez değerlendirilir.
- Geçersiz oranlar bahis hesabına dahil edilmez.
"""

import sys
import warnings
from collections import deque
import numpy as np
import pandas as pd

from diagnostics import run_diagnostics

from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. AYARLAR
# ============================================================

THRESHOLD = 2.5

N_TRAIN = 8500
N_VAL = 1500

EDGE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)

MIN_TEAM_HISTORY = 3
MIN_CALIBRATION_ROWS = 100
MIN_BETS_FOR_SELECTION = 50

RANDOM_SEED = 42


# ============================================================
# 2. VERİ KONTROLÜ
# ============================================================

def require_columns(df):

    required = [
        "Date",
        "League",
        "HomeTeam",
        "AwayTeam",
        "Home_xG",
        "Away_xG",
        "FTHG",
        "FTAG",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Eksik sütunlar: " + ", ".join(missing)
        )


# ============================================================
# 3. PİYASA ORANLARI
# ============================================================


def valid_odds(series):

    values = np.array(
        pd.to_numeric(series, errors="coerce"),
        dtype=float,
        copy=True
    )

    values[
        ~np.isfinite(values) | (values <= 1.0)
    ] = np.nan

    return values



def market_data(df):

    # Strict temporal rule: only non-C odds fields are eligible here.
    # Closing-price columns (AvgC/B365C/MaxC/PC and similar) must never
    # be read as prediction-time features or fallback prices.

    n = len(df)

    avg_over = (
        valid_odds(df["odds_Avg>2.5"])
        if "odds_Avg>2.5" in df.columns
        else np.full(n, np.nan)
    )

    avg_under = (
        valid_odds(df["odds_Avg<2.5"])
        if "odds_Avg<2.5" in df.columns
        else np.full(n, np.nan)
    )

    b365_over = (
        valid_odds(df["odds_B365>2.5"])
        if "odds_B365>2.5" in df.columns
        else np.full(n, np.nan)
    )

    b365_under = (
        valid_odds(df["odds_B365<2.5"])
        if "odds_B365<2.5" in df.columns
        else np.full(n, np.nan)
    )

    # Öncelik Avg. İki taraf da geçerli değilse
    # B365 çiftine birlikte geçilir.
    avg_valid = (
        np.isfinite(avg_over)
        & np.isfinite(avg_under)
    )

    b365_valid = (
        np.isfinite(b365_over)
        & np.isfinite(b365_under)
    )

    over = np.where(
        avg_valid,
        avg_over,
        np.where(b365_valid, b365_over, np.nan)
    )

    under = np.where(
        avg_valid,
        avg_under,
        np.where(b365_valid, b365_under, np.nan)
    )

    valid = (
        np.isfinite(over)
        & np.isfinite(under)
        & (over > 1.0)
        & (under > 1.0)
    )

    p_market = np.full(n, 0.5, dtype=float)

    inv_over = 1.0 / over[valid]
    inv_under = 1.0 / under[valid]

    denominator = inv_over + inv_under

    p_market[valid] = (
        inv_over / denominator
    )

    return p_market, over, under, valid


# ============================================================
# 4. GEÇMİŞE DAYALI POISSON ÖZELLİKLERİ
# ============================================================

def _recent_average(history, value_index: int, window: int, fallback: float) -> float:
    sample = history[-window:]
    if not sample:
        return float(fallback)
    return float(np.mean([item[value_index] for item in sample]))


def _recent_average(history, value_index: int, window: int, fallback: float) -> float:
    sample = history[-window:]
    if not sample:
        return float(fallback)
    return float(np.mean([item[value_index] for item in sample]))


def _recent_expected_goals(
    home_for: float,
    home_against: float,
    away_for: float,
    away_against: float,
    league_average: float,
    home_advantage: float,
) -> tuple[float, float]:
    league_average = max(float(league_average), 0.05)
    lambda_home_recent = (
        (home_for / league_average)
        * (away_against / league_average)
        * league_average
        * home_advantage
    )
    lambda_away_recent = (
        (away_for / league_average)
        * (home_against / league_average)
        * league_average
    )
    return (
        float(np.clip(lambda_home_recent, 0.05, 5.0)),
        float(np.clip(lambda_away_recent, 0.05, 5.0)),
    )


def add_leakage_safe_features(df):

    """
    Her maç için lambda değerleri sadece daha eski
    tarihlerdeki maçların xG verilerinden hesaplanır.

    Aynı tarih grubundaki maçlar önce tahmin edilir.
    Ardından o günün maçları geçmiş istatistiklerine eklenir.
    """

    out = df.copy()

    n = len(out)

    home = out["HomeTeam"].astype(str).to_numpy()
    away = out["AwayTeam"].astype(str).to_numpy()
    league = out["League"].astype(str).to_numpy()

    # All fixtures on the same UTC calendar day must share the same
    # pre-day history, even when their kickoff timestamps differ.
    dates = pd.to_datetime(
        out["Date"],
        utc=True,
        errors="raise"
    ).dt.normalize().to_numpy()

    home_xg = out["Home_xG"].to_numpy(dtype=float)
    away_xg = out["Away_xG"].to_numpy(dtype=float)
    home_goals = out["FTHG"].to_numpy(dtype=float)
    away_goals = out["FTAG"].to_numpy(dtype=float)

    # (Lig, takım): xG-for, xG-against, maç sayısı
    team_stats = {}

    # Lig: toplam xG, takım-maç gözlem sayısı
    league_stats = {}

    # Actual-goal history is isolated by UTC calendar day as well.
    team_goal_stats = {}
    league_goal_stats = {}
    global_goal_stats = [0.0, 0]
    home_goal_stats = [0.0, 0]
    away_goal_stats = [0.0, 0]

    global_stats = [0.0, 0]

    home_stats = [0.0, 0]
    away_stats = [0.0, 0]

    # Last-15 xG history is updated only after all fixtures of a day
    # have received their predictions.
    team_recent = {}

    p_poisson = np.full(n, 0.5)
    p_poisson_goals = np.full(n, 0.5)

    lambda_home = np.full(n, np.nan)
    lambda_home_goals = np.full(n, np.nan)
    lambda_away_goals = np.full(n, np.nan)
    lambda_away = np.full(n, np.nan)
    home_for_mean = np.full(n, np.nan)
    home_against_mean = np.full(n, np.nan)
    away_for_mean = np.full(n, np.nan)
    away_against_mean = np.full(n, np.nan)
    home_recent_for = np.full(n, np.nan)
    home_recent_against = np.full(n, np.nan)
    away_recent_for = np.full(n, np.nan)
    away_recent_against = np.full(n, np.nan)
    home_recent_for_10 = np.full(n, np.nan)
    home_recent_against_10 = np.full(n, np.nan)
    away_recent_for_10 = np.full(n, np.nan)
    away_recent_against_10 = np.full(n, np.nan)
    home_recent_for_15 = np.full(n, np.nan)
    home_recent_against_15 = np.full(n, np.nan)
    away_recent_for_15 = np.full(n, np.nan)
    away_recent_against_15 = np.full(n, np.nan)
    lambda_home_recent_5 = np.full(n, np.nan)
    lambda_away_recent_5 = np.full(n, np.nan)
    lambda_total_recent_5 = np.full(n, np.nan)
    p_poisson_recent_5 = np.full(n, 0.5)
    lambda_home_recent_10 = np.full(n, np.nan)
    lambda_away_recent_10 = np.full(n, np.nan)
    lambda_total_recent_10 = np.full(n, np.nan)
    p_poisson_recent_10 = np.full(n, 0.5)
    lambda_home_recent_15 = np.full(n, np.nan)
    lambda_away_recent_15 = np.full(n, np.nan)
    lambda_total_recent_15 = np.full(n, np.nan)
    p_poisson_recent_15 = np.full(n, 0.5)
    league_average_feature = np.full(n, np.nan)
    home_advantage_feature = np.full(n, np.nan)
    home_history_count = np.zeros(n, dtype=float)
    away_history_count = np.zeros(n, dtype=float)

    date_groups = pd.Series(
        np.arange(n)
    ).groupby(
        dates,
        sort=False
    )

    for _, positions in date_groups:

        indexes = positions.to_numpy(dtype=int)

        # ----------------------------------------------------
        # Önce o tarihteki tüm maçların tahminleri
        # ----------------------------------------------------

        for k in indexes:

            lg = league[k]

            league_record = league_stats.get(
                lg,
                [0.0, 0]
            )

            if league_record[1] > 0:

                league_avg = (
                    league_record[0]
                    / league_record[1]
                )

            elif global_stats[1] > 0:

                league_avg = (
                    global_stats[0]
                    / global_stats[1]
                )

            else:

                league_avg = 1.25

            league_avg = max(
                float(league_avg),
                0.05
            )

            home_key = (lg, home[k])
            away_key = (lg, away[k])

            goal_league_record = league_goal_stats.get(lg, [0.0, 0])
            if goal_league_record[1] > 0:
                goal_league_avg = goal_league_record[0] / goal_league_record[1]
            elif global_goal_stats[1] > 0:
                goal_league_avg = global_goal_stats[0] / global_goal_stats[1]
            else:
                goal_league_avg = 1.25
            goal_league_avg = max(float(goal_league_avg), 0.05)

            home_goal_record = team_goal_stats.get(home_key, [0.0, 0.0, 0])
            away_goal_record = team_goal_stats.get(away_key, [0.0, 0.0, 0])

            if home_goal_record[2] >= MIN_TEAM_HISTORY:
                home_goal_for = home_goal_record[0] / home_goal_record[2]
                home_goal_against = home_goal_record[1] / home_goal_record[2]
            else:
                home_goal_for = goal_league_avg
                home_goal_against = goal_league_avg

            if away_goal_record[2] >= MIN_TEAM_HISTORY:
                away_goal_for = away_goal_record[0] / away_goal_record[2]
                away_goal_against = away_goal_record[1] / away_goal_record[2]
            else:
                away_goal_for = goal_league_avg
                away_goal_against = goal_league_avg

            if home_goal_stats[1] > 0:
                home_goal_avg = home_goal_stats[0] / home_goal_stats[1]
            else:
                home_goal_avg = goal_league_avg

            if away_goal_stats[1] > 0:
                away_goal_avg = away_goal_stats[0] / away_goal_stats[1]
            else:
                away_goal_avg = goal_league_avg

            home_goal_advantage = np.clip(
                home_goal_avg / max(away_goal_avg, 0.05),
                0.75,
                1.35,
            )

            lam_h_goals = (
                (home_goal_for / goal_league_avg)
                * (away_goal_against / goal_league_avg)
                * goal_league_avg
                * home_goal_advantage
            )
            lam_a_goals = (
                (away_goal_for / goal_league_avg)
                * (home_goal_against / goal_league_avg)
                * goal_league_avg
            )
            lam_h_goals = float(np.clip(lam_h_goals, 0.05, 5.0))
            lam_a_goals = float(np.clip(lam_a_goals, 0.05, 5.0))

            lambda_home_goals[k] = lam_h_goals
            lambda_away_goals[k] = lam_a_goals
            p_poisson_goals[k] = poisson.sf(
                2,
                lam_h_goals + lam_a_goals,
            )

            hs = team_stats.get(
                home_key,
                [0.0, 0.0, 0]
            )

            aws = team_stats.get(
                away_key,
                [0.0, 0.0, 0]
            )

            if hs[2] >= MIN_TEAM_HISTORY:

                home_for = hs[0] / hs[2]
                home_against = hs[1] / hs[2]

            else:

                home_for = league_avg
                home_against = league_avg

            if aws[2] >= MIN_TEAM_HISTORY:

                away_for = aws[0] / aws[2]
                away_against = aws[1] / aws[2]

            else:

                away_for = league_avg
                away_against = league_avg

            if home_stats[1] > 0:

                home_avg = (
                    home_stats[0]
                    / home_stats[1]
                )

            else:

                home_avg = league_avg

            if away_stats[1] > 0:

                away_avg = (
                    away_stats[0]
                    / away_stats[1]
                )

            else:

                away_avg = league_avg

            home_adv = np.clip(
                home_avg / max(away_avg, 0.05),
                0.75,
                1.35
            )

            home_recent = list(team_recent.get(home_key, ()))
            away_recent = list(team_recent.get(away_key, ()))

            recent_home_for = _recent_average(home_recent, 0, 5, home_for)
            recent_home_against = _recent_average(home_recent, 1, 5, home_against)
            recent_away_for = _recent_average(away_recent, 0, 5, away_for)
            recent_away_against = _recent_average(away_recent, 1, 5, away_against)

            recent_home_for_10 = _recent_average(home_recent, 0, 10, home_for)
            recent_home_against_10 = _recent_average(home_recent, 1, 10, home_against)
            recent_away_for_10 = _recent_average(away_recent, 0, 10, away_for)
            recent_away_against_10 = _recent_average(away_recent, 1, 10, away_against)
            recent_home_for_15 = _recent_average(home_recent, 0, 15, home_for)
            recent_home_against_15 = _recent_average(home_recent, 1, 15, home_against)
            recent_away_for_15 = _recent_average(away_recent, 0, 15, away_for)
            recent_away_against_15 = _recent_average(away_recent, 1, 15, away_against)

            home_recent_for_10[k] = recent_home_for_10
            home_recent_against_10[k] = recent_home_against_10
            away_recent_for_10[k] = recent_away_for_10
            away_recent_against_10[k] = recent_away_against_10
            home_recent_for_15[k] = recent_home_for_15
            home_recent_against_15[k] = recent_home_against_15
            away_recent_for_15[k] = recent_away_for_15
            away_recent_against_15[k] = recent_away_against_15

            for window, values in (
                (5, (recent_home_for, recent_home_against, recent_away_for, recent_away_against)),
                (10, (recent_home_for_10, recent_home_against_10, recent_away_for_10, recent_away_against_10)),
                (15, (recent_home_for_15, recent_home_against_15, recent_away_for_15, recent_away_against_15)),
            ):
                recent_lambda_h, recent_lambda_a = _recent_expected_goals(
                    *values,
                    league_avg,
                    home_adv,
                )
                recent_lambda_total = recent_lambda_h + recent_lambda_a
                recent_probability = poisson.sf(2, recent_lambda_total)
                if window == 5:
                    lambda_home_recent_5[k] = recent_lambda_h
                    lambda_away_recent_5[k] = recent_lambda_a
                    lambda_total_recent_5[k] = recent_lambda_total
                    p_poisson_recent_5[k] = recent_probability
                elif window == 10:
                    lambda_home_recent_10[k] = recent_lambda_h
                    lambda_away_recent_10[k] = recent_lambda_a
                    lambda_total_recent_10[k] = recent_lambda_total
                    p_poisson_recent_10[k] = recent_probability
                else:
                    lambda_home_recent_15[k] = recent_lambda_h
                    lambda_away_recent_15[k] = recent_lambda_a
                    lambda_total_recent_15[k] = recent_lambda_total
                    p_poisson_recent_15[k] = recent_probability

            home_for_mean[k] = home_for
            home_against_mean[k] = home_against
            away_for_mean[k] = away_for
            away_against_mean[k] = away_against
            home_recent_for[k] = recent_home_for
            home_recent_against[k] = recent_home_against
            away_recent_for[k] = recent_away_for
            away_recent_against[k] = recent_away_against
            league_average_feature[k] = league_avg
            home_advantage_feature[k] = home_adv
            home_history_count[k] = hs[2]
            away_history_count[k] = aws[2]

            lam_h = (
                (home_for / league_avg)
                * (away_against / league_avg)
                * league_avg
                * home_adv
            )

            lam_a = (
                (away_for / league_avg)
                * (home_against / league_avg)
                * league_avg
            )

            lam_h = float(
                np.clip(lam_h, 0.05, 5.0)
            )

            lam_a = float(
                np.clip(lam_a, 0.05, 5.0)
            )

            lambda_home[k] = lam_h
            lambda_away[k] = lam_a

            # Bağımsız Poisson toplamı:
            # P(toplam gol > 2.5) = P(toplam gol >= 3)
            p_poisson[k] = poisson.sf(
                2,
                lam_h + lam_a
            )

        # ----------------------------------------------------
        # Tahminler tamamlandı.
        # Şimdi o tarihin gerçekleşmiş xG verilerini ekle.
        # ----------------------------------------------------

        for k in indexes:

            hg = home_goals[k]
            ag = away_goals[k]
            lg = league[k]

            if np.isfinite(hg) and np.isfinite(ag):
                home_key = (lg, home[k])
                away_key = (lg, away[k])

                home_goal_record = team_goal_stats.setdefault(
                    home_key, [0.0, 0.0, 0]
                )
                away_goal_record = team_goal_stats.setdefault(
                    away_key, [0.0, 0.0, 0]
                )

                home_goal_record[0] += hg
                home_goal_record[1] += ag
                home_goal_record[2] += 1

                away_goal_record[0] += ag
                away_goal_record[1] += hg
                away_goal_record[2] += 1

                league_goal_record = league_goal_stats.setdefault(
                    lg, [0.0, 0]
                )
                league_goal_record[0] += hg + ag
                league_goal_record[1] += 2

                global_goal_stats[0] += hg + ag
                global_goal_stats[1] += 2
                home_goal_stats[0] += hg
                home_goal_stats[1] += 1
                away_goal_stats[0] += ag
                away_goal_stats[1] += 1

            hxg = home_xg[k]
            axg = away_xg[k]

            if not (
                np.isfinite(hxg)
                and np.isfinite(axg)
            ):
                continue


            home_key = (lg, home[k])
            away_key = (lg, away[k])

            hs = team_stats.setdefault(
                home_key,
                [0.0, 0.0, 0]
            )

            aws = team_stats.setdefault(
                away_key,
                [0.0, 0.0, 0]
            )

            # Ev sahibi takım
            hs[0] += hxg
            hs[1] += axg
            hs[2] += 1

            # Deplasman takımı
            aws[0] += axg
            aws[1] += hxg
            aws[2] += 1

            team_recent.setdefault(
                home_key,
                deque(maxlen=15)
            ).append((hxg, axg))

            team_recent.setdefault(
                away_key,
                deque(maxlen=15)
            ).append((axg, hxg))

            ls = league_stats.setdefault(
                lg,
                [0.0, 0]
            )

            ls[0] += hxg + axg
            ls[1] += 2

            global_stats[0] += hxg + axg
            global_stats[1] += 2

            home_stats[0] += hxg
            home_stats[1] += 1

            away_stats[0] += axg
            away_stats[1] += 1

    out["p_poisson"] = np.clip(
        p_poisson,
        1e-6,
        1 - 1e-6
    )
    out["p_poisson_goals"] = np.clip(
        p_poisson_goals,
        1e-6,
        1 - 1e-6
    )

    out["lambda_home_goals"] = lambda_home_goals
    out["lambda_away_goals"] = lambda_away_goals
    out["lambda_home"] = lambda_home
    out["lambda_away"] = lambda_away
    out["home_for_mean"] = home_for_mean
    out["home_against_mean"] = home_against_mean
    out["away_for_mean"] = away_for_mean
    out["away_against_mean"] = away_against_mean
    out["home_recent_for"] = home_recent_for
    out["home_recent_against"] = home_recent_against
    out["away_recent_for"] = away_recent_for
    out["away_recent_against"] = away_recent_against
    out["home_recent_for_10"] = home_recent_for_10
    out["home_recent_against_10"] = home_recent_against_10
    out["away_recent_for_10"] = away_recent_for_10
    out["away_recent_against_10"] = away_recent_against_10
    out["home_recent_for_15"] = home_recent_for_15
    out["home_recent_against_15"] = home_recent_against_15
    out["away_recent_for_15"] = away_recent_for_15
    out["away_recent_against_15"] = away_recent_against_15
    out["lambda_home_recent_5"] = lambda_home_recent_5
    out["lambda_away_recent_5"] = lambda_away_recent_5
    out["lambda_total_recent_5"] = lambda_total_recent_5
    out["p_poisson_recent_5"] = np.clip(p_poisson_recent_5, 1e-6, 1 - 1e-6)
    out["lambda_home_recent_10"] = lambda_home_recent_10
    out["lambda_away_recent_10"] = lambda_away_recent_10
    out["lambda_total_recent_10"] = lambda_total_recent_10
    out["p_poisson_recent_10"] = np.clip(p_poisson_recent_10, 1e-6, 1 - 1e-6)
    out["lambda_home_recent_15"] = lambda_home_recent_15
    out["lambda_away_recent_15"] = lambda_away_recent_15
    out["lambda_total_recent_15"] = lambda_total_recent_15
    out["p_poisson_recent_15"] = np.clip(p_poisson_recent_15, 1e-6, 1 - 1e-6)
    out["league_average"] = league_average_feature
    out["home_advantage"] = home_advantage_feature
    out["home_history_count"] = home_history_count
    out["away_history_count"] = away_history_count

    return out


# ============================================================
# 5. BLEND MODEL
# ============================================================

class BlendModel:

    def __init__(self):

        self.scaler = StandardScaler()

        self.clf = LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_SEED
        )

        self.iso = None

        self.is_calibrated = False

    @staticmethod
    def _X(df):

        p_market, _, _, market_valid = market_data(df)

        p_poisson = df[
            "p_poisson"
        ].to_numpy(dtype=float)

        X = np.column_stack([
            p_market,
            p_poisson,
            market_valid.astype(float),
        ])

        return np.nan_to_num(
            X,
            nan=0.5,
            posinf=0.5,
            neginf=0.5
        )

    @staticmethod
    def _y(df):

        total_goals = (
            df["FTHG"].to_numpy(dtype=float)
            + df["FTAG"].to_numpy(dtype=float)
        )

        return (
            total_goals > THRESHOLD
        ).astype(int)

    def fit(self, df):

        X = self._X(df)
        y = self._y(df)

        self.scaler.fit(X)

        self.clf.fit(
            self.scaler.transform(X),
            y
        )

        return self

    def raw_predict(self, df):

        X = self._X(df)

        return self.clf.predict_proba(
            self.scaler.transform(X)
        )[:, 1]

    def calibrate(self, df):

        if len(df) < MIN_CALIBRATION_ROWS:

            raise ValueError(
                "Kalibrasyon verisi çok küçük."
            )

        y = self._y(df)

        if np.unique(y).size < 2:

            raise ValueError(
                "Kalibrasyon diliminde iki sınıf da bulunmalı."
            )

        p = self.raw_predict(df)

        self.iso = IsotonicRegression(
            y_min=0.001,
            y_max=0.999,
            out_of_bounds="clip"
        )

        self.iso.fit(p, y)

        self.is_calibrated = True

        return self

    def predict_proba(self, df):

        p = self.raw_predict(df)

        if self.is_calibrated:

            p = self.iso.predict(p)

        return np.clip(
            p,
            1e-6,
            1 - 1e-6
        )


class HistoricalXGModel(BlendModel):
    """Prediction model using only strictly historical xG-derived features.

    Bookmaker odds are intentionally excluded until the source and exact
    prediction-time availability of every odds column are independently proven.
    """

    @staticmethod
    def _X(df):
        required = (
            "lambda_home",
            "lambda_away",
            "p_poisson",
            "home_for_mean",
            "home_against_mean",
            "away_for_mean",
            "away_against_mean",
            "home_recent_for",
            "home_recent_against",
            "away_recent_for",
            "away_recent_against",
            "home_recent_for_10",
            "home_recent_against_10",
            "away_recent_for_10",
            "away_recent_against_10",
            "home_recent_for_15",
            "home_recent_against_15",
            "away_recent_for_15",
            "away_recent_against_15",
            "lambda_home_recent_5",
            "lambda_away_recent_5",
            "lambda_total_recent_5",
            "p_poisson_recent_5",
            "lambda_home_recent_10",
            "lambda_away_recent_10",
            "lambda_total_recent_10",
            "p_poisson_recent_10",
            "lambda_home_recent_15",
            "lambda_away_recent_15",
            "lambda_total_recent_15",
            "p_poisson_recent_15",
            "league_average",
            "home_advantage",
            "home_history_count",
            "away_history_count",
        )
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                "Historical XG features missing: " + ", ".join(missing)
            )

        X = df.loc[:, required].to_numpy(dtype=float)

        return np.nan_to_num(
            X,
            nan=1.25,
            posinf=5.0,
            neginf=0.05,
        )


class PoissonOnlyModel:
    """Prediction-time model based only on leakage-safe historical xG Poisson rates."""

    @staticmethod
    def _y(df):
        return BlendModel._y(df)

    def fit(self, df):
        # No outcome-fitted coefficients: all signal is from prior-match xG rates.
        return self

    def calibrate(self, df):
        # Raw Poisson probabilities outperformed the fitted calibrators in
        # expanding-window development folds; do not recalibrate on this data.
        return self

    @staticmethod
    def predict_proba(df):
        if "p_poisson" not in df.columns:
            raise ValueError("Missing leakage-safe p_poisson feature.")
        return np.clip(
            df["p_poisson"].to_numpy(dtype=float),
            1e-6,
            1.0 - 1e-6,
        )

    @staticmethod
    def raw_predict(df):
        return PoissonOnlyModel.predict_proba(df)


# ============================================================
# 6. KALİBRASYON METRİĞİ
# ============================================================

def expected_calibration_error(y, p, bins=10):

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1
    )

    ece = 0.0

    for i in range(bins):

        if i < bins - 1:

            mask = (
                (p >= edges[i])
                & (p < edges[i + 1])
            )

        else:

            mask = (
                (p >= edges[i])
                & (p <= edges[i + 1])
            )

        if mask.any():

            observed = y[mask].mean()
            predicted = p[mask].mean()

            ece += (
                mask.mean()
                * abs(observed - predicted)
            )

    return float(ece)


# ============================================================
# 7. PERFORMANS RAPORU
# ============================================================

def evaluate(name, df, model):

    p = model.predict_proba(df)

    y = model._y(df)

    pred = (
        p >= 0.5
    ).astype(int)

    print("\n" + "=" * 70)

    print(
        f"{name} — MAÇ SAYISI: {len(y)}"
    )

    print("=" * 70)

    print(
        f"OVER oranı : {y.mean():.3f}"
    )

    print(
        f"Accuracy   : {accuracy_score(y, pred):.4f}"
    )

    print(
        f"Log Loss   : {log_loss(y, p, labels=[0, 1]):.4f}"
    )

    print(
        f"Brier      : {brier_score_loss(y, p):.4f}"
    )

    print(
        f"ECE        : {expected_calibration_error(y, p):.4f}"
    )

    if np.unique(y).size == 2:

        print(
            f"AUC        : {roc_auc_score(y, p):.4f}"
        )

    else:

        print(
            "AUC        : hesaplanamadı"
        )

    print(
        classification_report(
            y,
            pred,
            labels=[0, 1],
            target_names=["UNDER", "OVER"],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    )

    print(
        pd.DataFrame(
            cm,
            index=[
                "Gerçek UNDER",
                "Gerçek OVER"
            ],
            columns=[
                "Tahmin UNDER",
                "Tahmin OVER"
            ]
        ).to_string()
    )


# ============================================================
# 8. VALUE BET ANALİZİ
# ============================================================

def confidence_policy_analysis(df, model, label, cutoff=0.75):
    """Evaluate a fixed, odds-independent confidence gate; no threshold tuning."""
    y = model._y(df)
    probabilities = model.predict_proba(df)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    selected = confidence >= float(cutoff)

    count = int(selected.sum())
    total = int(len(df))
    predicted = (probabilities >= 0.5).astype(int)
    correct = predicted == y

    print("\\n" + "=" * 78)
    print(f"CONFIDENCE POLICY — {label} — FIXED CUTOFF: {cutoff:.2f}")
    print("=" * 78)
    print(f"Selected picks : {count} / {total}")
    print(f"Coverage       : {count / total:.1%}" if total else "Coverage       : 0.0%")

    if count == 0:
        print("Accuracy       : NO PICK")
        return {
            "label": label,
            "cutoff": float(cutoff),
            "matches": 0,
            "accuracy": np.nan,
            "coverage": 0.0,
        }

    accuracy = float(correct[selected].mean())
    over_mask = selected & (predicted == 1)
    under_mask = selected & (predicted == 0)

    selected_indices = np.flatnonzero(selected)
    selected_order = np.argsort(confidence[selected_indices])[::-1]
    top_indices = selected_indices[selected_order[:10]]
    top_correct = int(correct[top_indices].sum())
    top_accuracy = float(correct[top_indices].mean()) if len(top_indices) else np.nan

    print(f"Correct        : {int(correct[selected].sum())} / {count}")
    print(f"Accuracy       : {accuracy:.2%}")
    print(
        f"Top-10 picks   : {top_correct} / {len(top_indices)} "
        f"| accuracy: {top_accuracy:.2%}"
    )
    print(
        f"OVER picks     : {int(over_mask.sum())} "
        f"| hit rate: {float(correct[over_mask].mean()):.2%}"
        if over_mask.any() else "OVER picks     : 0"
    )
    print(
        f"UNDER picks    : {int(under_mask.sum())} "
        f"| hit rate: {float(correct[under_mask].mean()):.2%}"
        if under_mask.any() else "UNDER picks    : 0"
    )

    return {
        "label": label,
        "cutoff": float(cutoff),
        "matches": count,
        "correct": int(correct[selected].sum()),
        "accuracy": accuracy,
        "top_n": int(len(top_indices)),
        "top_n_correct": top_correct,
        "top_n_accuracy": top_accuracy,
        "coverage": float(count / total) if total else 0.0,
    }


def value_bet_analysis(
    df,
    model,
    label,
    thresholds=EDGE_THRESHOLDS
):

    p_model = model.predict_proba(df)

    (
        p_market,
        over_odds,
        under_odds,
        market_valid
    ) = market_data(df)

    y = model._y(df)

    edge = p_model - p_market

    print("\n" + "=" * 78)

    print(
        f"VALUE BET ANALİZİ — {label}"
    )

    print("=" * 78)

    print(
        f"Geçerli oranlı maç: "
        f"{market_valid.sum()} / {len(df)}"
    )

    if market_valid.any():

        print(
            f"Ortalama edge: "
            f"{edge[market_valid].mean():+.4f}"
        )

        print(
            f"Edge std    : "
            f"{edge[market_valid].std():.4f}"
        )

        for t in [0.03, 0.05]:

            count = (
                market_valid
                & (np.abs(edge) > t)
            ).sum()

            ratio = (
                count / market_valid.sum()
            )

            print(
                f"|edge| > {t:.2f}: "
                f"{count} maç (%{ratio * 100:.1f})"
            )

    print()

    print(
        f"{'Eşik':>7} "
        f"{'Bahis':>8} "
        f"{'İsabet':>9} "
        f"{'Ort.Oran':>9} "
        f"{'ROI %':>9} "
        f"{'Net Kâr':>11}"
    )

    print("-" * 65)

    results = []

    for threshold in thresholds:

        over_idx = np.flatnonzero(
            market_valid
            & (edge > threshold)
            & np.isfinite(over_odds)
            & (over_odds > 1.0)
        )

        under_idx = np.flatnonzero(
            market_valid
            & (edge < -threshold)
            & np.isfinite(under_odds)
            & (under_odds > 1.0)
        )

        profits = []
        odds_used = []

        wins = 0

        for idx in over_idx:

            won = (
                y[idx] == 1
            )

            wins += int(won)

            if won:

                profits.append(
                    over_odds[idx] - 1.0
                )

            else:

                profits.append(-1.0)

            odds_used.append(
                over_odds[idx]
            )

        for idx in under_idx:

            won = (
                y[idx] == 0
            )

            wins += int(won)

            if won:

                profits.append(
                    under_odds[idx] - 1.0
                )

            else:

                profits.append(-1.0)

            odds_used.append(
                under_odds[idx]
            )

        n_bets = len(profits)

        if n_bets == 0:

            print(
                f"{threshold:>7.2f} "
                f"{0:>8}"
            )

            results.append({
                "threshold": threshold,
                "bets": 0,
                "hit": np.nan,
                "avg_odds": np.nan,
                "roi": np.nan,
                "profit": 0.0
            })

            continue

        profits = np.asarray(
            profits,
            dtype=float
        )

        total_profit = float(
            profits.sum()
        )

        roi = (
            total_profit / n_bets
        ) * 100.0

        hit = (
            wins / n_bets
        )

        avg_odds = float(
            np.mean(odds_used)
        )

        print(
            f"{threshold:>7.2f} "
            f"{n_bets:>8d} "
            f"{hit:>8.1%} "
            f"{avg_odds:>9.2f} "
            f"{roi:>+8.2f} "
            f"{total_profit:>+11.2f}"
        )

        results.append({
            "threshold": threshold,
            "bets": n_bets,
            "hit": hit,
            "avg_odds": avg_odds,
            "roi": roi,
            "profit": total_profit
        })

    return pd.DataFrame(results)


# ============================================================
# 9. EŞİK SEÇİMİ
# ============================================================

def select_edge_threshold(results):

    eligible = results[
        (results["bets"] >= MIN_BETS_FOR_SELECTION)
        & np.isfinite(results["roi"])
    ]

    if eligible.empty:

        return None

    best = eligible.sort_values(
        ["roi", "bets"],
        ascending=[False, False]
    ).iloc[0]

    return float(
        best["threshold"]
    )


# ============================================================
# 10. ANA AKIŞ
# ============================================================

def main(path, skip_final_test=False):

    print("\nCSV okunuyor...")

    df = pd.read_csv(
        path,
        encoding="latin-1",
        low_memory=False
    )

    require_columns(df)

    df["Date"] = pd.to_datetime(
        df["Date"],
        utc=True,
        errors="coerce"
    )

    numeric_columns = [
        "Home_xG",
        "Away_xG",
        "FTHG",
        "FTAG"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Date",
            "League",
            "HomeTeam",
            "AwayTeam",
            "Home_xG",
            "Away_xG",
            "FTHG",
            "FTAG"
        ]
    ).copy()

    df = df.sort_values(
        [
            "Date",
            "League",
            "HomeTeam",
            "AwayTeam"
        ],
        kind="mergesort"
    ).reset_index(drop=True)

    total = len(df)

    if total <= N_TRAIN + N_VAL + 100:

        raise ValueError(
            f"Yetersiz veri: {total} maç."
        )

    print(
        f"Toplam maç: {total}"
    )

    # Policy-only CI must not load, transform, or inspect the
    # sealed final-test rows. Keep only train + validation rows
    # before chronological feature generation.
    if skip_final_test:
        df = df.iloc[:N_TRAIN + N_VAL].copy()

    # --------------------------------------------------------
    # Özellik üretimi
    # --------------------------------------------------------

    print(
        "\nGeçmişe dayalı Poisson özellikleri hesaplanıyor..."
    )

    df = add_leakage_safe_features(df)

    # --------------------------------------------------------
    # Kronolojik ana bölme
    # --------------------------------------------------------

    train = df.iloc[
        :N_TRAIN
    ].copy()

    validation = df.iloc[
        N_TRAIN:N_TRAIN + N_VAL
    ].copy()

    test = None if skip_final_test else df.iloc[
        N_TRAIN + N_VAL:
    ].copy()

    # --------------------------------------------------------
    # Onay diliminin üç parçaya ayrılması
    # --------------------------------------------------------

    n_cal = int(
        len(validation) * 0.60
    )

    n_select = int(
        len(validation) * 0.20
    )

    calibration = validation.iloc[
        :n_cal
    ].copy()

    policy_select = validation.iloc[
        n_cal:n_cal + n_select
    ].copy()

    policy_check = validation.iloc[
        n_cal + n_select:
    ].copy()

    print("\n" + "=" * 70)

    print("KRONOLOJİK VERİ BÖLÜMÜ")

    print("=" * 70)

    parts = [
        ("TRAIN", train),
        ("CALIBRATION", calibration),
        ("POLICY SELECT", policy_select),
        ("POLICY CHECK", policy_check),
    ]

    if test is not None:
        parts.append(("FINAL TEST", test))

    for name, part in parts:

        print(
            f"{name:15}: "
            f"{len(part):5d} maç | "
            f"{part.Date.min().date()} → "
            f"{part.Date.max().date()}"
        )

    # --------------------------------------------------------
    # Model eğitimi
    # --------------------------------------------------------

    print(
        "\nModel eğitiliyor..."
    )

    model = PoissonOnlyModel()

    model.fit(train)

    print(
        "Kalibrasyon yapılıyor..."
    )

    model.calibrate(calibration)

    # --------------------------------------------------------
    # Eğitim teşhisi
    # --------------------------------------------------------

    print(
        "\nUYARI: TRAIN sonuçları bağımsız performans değildir."
    )

    evaluate(
        "TRAIN — EĞİTİM İÇİ TEŞHİS",
        train,
        model
    )

    # --------------------------------------------------------
    # Bağımsız eşik seçimi
    # --------------------------------------------------------

    confidence_policy_analysis(
        policy_select,
        model,
        "POLICY SELECT",
        cutoff=0.70,
    )

    confidence_policy_analysis(
        policy_select,
        model,
        "POLICY SELECT",
        cutoff=0.75,
    )

    selection_results = value_bet_analysis(
        policy_select,
        model,
        "POLICY SELECT"
    )

    chosen_threshold = select_edge_threshold(
        selection_results
    )

    if chosen_threshold is None:

        print(
            "\nYeterli bahis sayısı bulunamadı."
        )

        print(
            "Bahis politikası: NO BET"
        )

    else:

        print(
            f"\nSeçilen edge eşiği: "
            f"{chosen_threshold:.2f}"
        )

    # --------------------------------------------------------
    # Eşik doğrulaması
    # --------------------------------------------------------

    evaluate(
        "POLICY CHECK — BAĞIMSIZ DOĞRULAMA",
        policy_check,
        model
    )

    confidence_policy_analysis(
        policy_check,
        model,
        "POLICY CHECK",
        cutoff=0.70,
    )

    confidence_policy_analysis(
        policy_check,
        model,
        "POLICY CHECK",
        cutoff=0.75,
    )

    if chosen_threshold is not None:

        value_bet_analysis(
            policy_check,
            model,
            "POLICY CHECK",
            thresholds=(chosen_threshold,)
        )

        run_diagnostics(
            df=policy_check,
            model=model,
            label="POLICY_CHECK",
            threshold=chosen_threshold,
            market_data_fn=market_data,
        )

    # --------------------------------------------------------
    # Final test is deliberately skipped by CI's policy-only mode.
    if skip_final_test:
        print(
            "\nPOLICY-ONLY MODE: FINAL TEST SEALED — NOT ACCESSED"
        )
    else:
        # Final test
        # --------------------------------------------------------

        print("\n" + "#" * 78)

        print(
            "FINAL TEST — TEK SEFERLİK BAĞIMSIZ DEĞERLENDİRME"
        )

        print("#" * 78)

        evaluate(
            "FINAL TEST",
            test,
            model
        )

        if chosen_threshold is not None:

            value_bet_analysis(
                test,
                model,
                "FINAL TEST",
                thresholds=(chosen_threshold,)
            )

            run_diagnostics(
                df=test,
                model=model,
                label="FINAL_TEST",
                threshold=chosen_threshold,
                market_data_fn=market_data,
            )

        else:

            print(
                "\nFINAL TEST BAHİS POLİTİKASI: NO BET"
            )


    print("\n" + "=" * 70)

    print(
        "ANALİZ TAMAMLANDI"
    )

    print("=" * 70)

    print(
        "Final test sonuçları model/eşik seçimi için kullanılmamalıdır."
    )


# ============================================================
# 11. ÇALIŞTIR
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) not in (2, 3) or (
        len(sys.argv) == 3 and sys.argv[2] != "--policy-only"
    ):

        print(
            "Kullanım: python BOT.py xgo.csv [--policy-only]"
        )

        sys.exit(1)

    try:

        main(
            sys.argv[1],
            skip_final_test=(len(sys.argv) == 3)
        )

    except Exception as exc:

        print(
            f"\nHATA: {exc}"
        )

        sys.exit(1)
