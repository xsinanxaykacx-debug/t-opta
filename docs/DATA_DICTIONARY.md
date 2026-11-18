# xg18.csv data dictionary

Source: repository-root `xg18.csv` (UTF-8 with BOM). The Historical Engine accepts the
default `data/raw/xg18.csv` path and falls back to the repository-root file.

Observed source profile: 18,509 rows, 39 columns, 18,505 valid timestamps, 4 blank dates,
15 leagues, seasons 2022/2023 through 2026/2027. The file contains no bookmaker odds.

| Column group | Fields | Use |
|---|---|---|
| Identity | MatchID, League, Season, Date, HomeTeam, AwayTeam, Referee | Match identity, filtering and chronology |
| Final score | FTHG, FTAG | Historical outcomes only |
| Half-time | HT_Home_Goals, HT_Away_Goals, HT_Home_xG, HT_Away_xG | Historical only; half-time xG missing in 2,753 rows per side |
| Expected goals | Home_xG, Away_xG | Historical match performance |
| Shots | Home/Away_Shots, ShotsInBox, ShotsSetPiece, ShotsOnTarget | Historical match performance |
| Chances | Home_BigChance, Away_BigChance | Historical match performance |
| Corners/possession | Home/Away_Corners, Home/Away_Poss | Historical match performance |
| Discipline | Home/Away_RedCards, Penalty_Awarded | Historical match performance |
| Squad context | Home/Away_Squad_Value_M, Home/Away_Injured_Key_Players | Context fields; missing values must remain missing |
| Weather/attendance | Weather_Temp, Weather_Rain, Attendance | Context fields; high missingness |

Missing-value counts observed in the source:
- Date: 4; Referee: 403.
- HT_Home_xG and HT_Away_xG: 2,753 each.
- Home_Squad_Value_M and Away_Squad_Value_M: 1 each.
- Weather_Temp and Weather_Rain: 12,102 each.
- Attendance: 427.
- Other columns: no missing values in the observed CSV snapshot.

Leakage rule: for a target fixture, only identity fields (MatchID, league, season, kickoff, teams)
may be read from its row. Its final score, xG, shots, corners, possession, cards, penalties,
weather and attendance are not copied into pre-match features. Historical values are available
only from strictly earlier dated matches. No missing odds or other values are fabricated.
