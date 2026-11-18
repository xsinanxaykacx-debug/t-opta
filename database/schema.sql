CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL,
 home_probability REAL NOT NULL, draw_probability REAL NOT NULL, away_probability REAL NOT NULL,
 decision TEXT NOT NULL, model_version TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS match_results (
 id INTEGER PRIMARY KEY AUTOINCREMENT, match_id TEXT NOT NULL UNIQUE,
 home_goals INTEGER NOT NULL, away_goals INTEGER NOT NULL, recorded_at TEXT NOT NULL
);
