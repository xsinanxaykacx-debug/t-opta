"""Run the frozen chronological Logistic Regression baseline on xg18.csv."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.feature_engine import FeatureEngine
from modules.historical_engine import HistoricalEngine
from models.logistic_baseline import LogisticBaseline
from validation.backtest import HistoricalBacktester


def season_start_year(value):
    text = str(value).strip()
    import re
    match = re.search(r"(20\d{2})\s*[/\-]\s*(?:20)?\d{2}", text)
    if not match:
        return None
    return int(match.group(1))


def main():
    history = HistoricalEngine(ROOT / "xg18.csv")
    data = history.data.copy()
    data = data.loc[
        data["Date"].notna()
        & data["FTHG"].notna()
        & data["FTAG"].notna()
    ].copy()
    data["_season_start"] = data["Season"].map(season_start_year)

    unknown = sorted(data.loc[data["_season_start"].isna(), "Season"].dropna().astype(str).unique())
    if unknown:
        raise ValueError(f"Unrecognized season labels; refusing implicit split: {unknown}")

    train = data.loc[data["_season_start"].isin([2022, 2023])].drop(columns="_season_start")
    calibration = data.loc[data["_season_start"].eq(2024)].drop(columns="_season_start")
    test = data.loc[data["_season_start"].isin([2025, 2026])].drop(columns="_season_start")
    if train.empty or calibration.empty or test.empty:
        raise ValueError(
            "Required chronological seasons missing. Expected train 2022/23-2023/24, "
            "calibration 2024/25, and test 2025/26-2026/27."
        )

    for name, part in (("train", train), ("calibration", calibration), ("test", test)):
        print(f"{name}: {len(part):,} matches | {part['Date'].min().date()} -> {part['Date'].max().date()}")

    engine = FeatureEngine(history)
    model = LogisticBaseline()
    report = HistoricalBacktester(engine, model).run_backtest(train, calibration, test)
    report["model_selection"] = {
        "selected_C": model.selected_c_,
        "internal_chronological_validation_log_loss": model.validation_scores_,
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": model.max_iter,
        "calibration": "temperature_scaling",
        "temperature": model.temperature_,
        "preprocessing": "train-fitted median imputation + standard scaling + one-hot encoding",
    }
    report["season_partitions"] = {
        "train": ["2022/23", "2023/24"],
        "calibration": ["2024/25"],
        "test": ["2025/26", "2026/27"],
    }
    output = ROOT / "reports" / "logistic_baseline_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()
