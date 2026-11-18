import sqlite3
from pathlib import Path

class Database:
    def __init__(self, path="data/t-opta.sqlite3"):
        self.path = path

    def connect(self):
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def initialize(self):
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)

    def save_prediction(self, match_id, probabilities, decision, model_version, created_at):
        values = [probabilities[k] for k in ("home", "draw", "away")]
        if any(not 0 <= x <= 1 for x in values) or abs(sum(values)-1.0) > 1e-6:
            raise ValueError("Probabilities must be in [0,1] and sum to 1.")
        with self.connect() as conn:
            cur = conn.execute("INSERT INTO predictions (match_id,home_probability,draw_probability,away_probability,decision,model_version,created_at) VALUES (?,?,?,?,?,?,?)",
                               (match_id, *values, decision, model_version, created_at))
            return int(cur.lastrowid)

    def save_result(self, match_id, home_goals, away_goals, recorded_at):
        if home_goals < 0 or away_goals < 0:
            raise ValueError("Goals cannot be negative.")
        with self.connect() as conn:
            conn.execute("INSERT INTO match_results (match_id,home_goals,away_goals,recorded_at) VALUES (?,?,?,?) ON CONFLICT(match_id) DO UPDATE SET home_goals=excluded.home_goals, away_goals=excluded.away_goals, recorded_at=excluded.recorded_at",
                         (match_id, home_goals, away_goals, recorded_at))
