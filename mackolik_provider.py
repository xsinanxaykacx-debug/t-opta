"""Fetch upcoming football fixtures and 1X2 odds directly from Mackolik.

The provider uses Mackolik's public football livescores JSON feed for fixture discovery
and the corresponding Mackolik match/Iddaa pages for the displayed match-result prices.
No third-party odds provider or manual upcoming CSV is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cloudscraper
from bs4 import BeautifulSoup

BASE_URL = "https://www.mackolik.com"
FIXTURES_URL = BASE_URL + "/perform/p0/ajax/components/competition/livescores/json"
LOCAL_TZ = ZoneInfo("Europe/Istanbul")
THREAD_LOCAL = threading.local()
CSV_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA",
    "League", "Source", "MatchId", "MatchURL", "OddsCapturedAt",
]


def _session():
    """One Cloudflare-capable HTTP session per worker thread."""
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        session.headers.update({
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.7",
            "Referer": BASE_URL + "/iddaa",
        })
        THREAD_LOCAL.session = session
    return session


def _get(url: str, *, params: dict | None = None, attempts: int = 3):
    last_error = None
    for attempt in range(attempts):
        try:
            response = _session().get(url, params=params, timeout=25)
            response.raise_for_status()
            return response
        except Exception as exc:  # network/provider errors are reported with context
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Maçkolik isteği başarısız: {url} ({last_error})") from last_error


def _local_kickoff(match: dict) -> datetime | None:
    raw = match.get("mstUtc")
    try:
        return datetime.fromtimestamp(float(raw) / 1000.0, tz=timezone.utc).astimezone(LOCAL_TZ)
    except (TypeError, ValueError, OSError):
        return None


def fetch_fixtures(days: int = 7, now: datetime | None = None) -> list[dict]:
    """Return future, pre-match football fixtures listed by Mackolik."""
    if days < 1 or days > 14:
        raise ValueError("Gün aralığı 1 ile 14 arasında olmalı.")
    now = now or datetime.now(LOCAL_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)

    fixtures: dict[str, dict] = {}
    for offset in range(days):
        match_day = (now + timedelta(days=offset)).date().isoformat()
        response = _get(
            FIXTURES_URL,
            params={"sports[]": "Soccer", "matchDate": match_day},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Maçkolik fikstür yanıtı JSON değil: {match_day}") from exc

        matches = payload.get("data", {}).get("matches", {})
        if not isinstance(matches, dict):
            continue
        for match_id, item in matches.items():
            if not isinstance(item, dict) or item.get("state") != "pre":
                continue
            kickoff = _local_kickoff(item)
            if kickoff is None or kickoff <= now:
                continue
            home = (item.get("homeTeam") or {}).get("name")
            away = (item.get("awayTeam") or {}).get("name")
            slug = item.get("matchSlug")
            # Maçkolik only lists an Iddaa detail page when an Iddaa code exists.
            if not home or not away or not slug or not item.get("iddaaCode"):
                continue
            fixtures[str(match_id)] = {
                "match_id": str(match_id),
                "match_slug": str(slug),
                "home": str(home).strip(),
                "away": str(away).strip(),
                "kickoff": kickoff,
                "competition_id": item.get("competitionId"),
                "league_name": item.get("competitionName") or item.get("tournamentName") or ((item.get("competition") or {}).get("name") if isinstance(item.get("competition"), dict) else None) or str(item.get("competitionId") or "Diğer ligler"),
                "iddaa_code": item.get("iddaaCode"),
            }
    return sorted(fixtures.values(), key=lambda row: row["kickoff"])


def _parse_price(value: str) -> float | None:
    cleaned = value.strip().replace(",", ".")
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return None
    price = float(cleaned)
    return price if price > 1.0 else None


def parse_match_odds(html: str) -> tuple[float, float, float] | None:
    """Read the first/current 1-X-2 row under Mackolik's Maç Sonucu market."""
    soup = BeautifulSoup(html, "html.parser")
    heading = next(
        (
            node for node in soup.find_all("h2")
            if "maç sonucu" in node.get_text(" ", strip=True).casefold()
        ),
        None,
    )
    if heading is None:
        return None

    market = heading.find_parent("li")
    if market is None:
        return None
    odds_list = market.find("ul")
    if odds_list is None:
        return None

    prices: dict[str, float] = {}
    for item in odds_list.find_all("li", recursive=False):
        parts = item.get_text(" ", strip=True).split()
        if len(parts) < 2:
            continue
        label = parts[0].upper()
        if label not in {"1", "X", "2"}:
            continue
        price = _parse_price(parts[-1])
        if price is not None:
            prices[label] = price

    if all(label in prices for label in ("1", "X", "2")):
        return prices["1"], prices["X"], prices["2"]
    return None


def fetch_match_odds(fixture: dict) -> dict | None:
    match_url = (
        f"{BASE_URL}/mac/{fixture['match_slug']}/iddaa/{fixture['match_id']}"
    )
    try:
        response = _get(match_url)
        odds = parse_match_odds(response.text)
        if odds is None:
            return None
        captured_at = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
        return {
            "Date": fixture["kickoff"].strftime("%Y-%m-%d %H:%M"),
            "HomeTeam": fixture["home"],
            "AwayTeam": fixture["away"],
            "AvgH": odds[0],
            "AvgD": odds[1],
            "AvgA": odds[2],
            "League": fixture.get("league_name", "Diğer ligler"),
            "Source": "Mackolik",
            "MatchId": fixture["match_id"],
            "MatchURL": match_url,
            "OddsCapturedAt": captured_at,
        }
    except Exception as exc:
        print(
            f"UYARI: {fixture['home']} - {fixture['away']} oranı alınamadı: {exc}",
            file=sys.stderr,
        )
        return None


def collect_upcoming(days: int = 7, workers: int = 3) -> list[dict]:
    fixtures = fetch_fixtures(days=days)
    print(f"Maçkolik fikstüründe {len(fixtures)} yaklaşan maç bulundu.")
    if not fixtures:
        return []

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 5))) as executor:
        futures = [executor.submit(fetch_match_odds, fixture) for fixture in fixtures]
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda row: row["Date"])
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Maçkolik'ten yaklaşan futbol maçlarını ve MS1/MSX/MS2 oranlarını al."
    )
    parser.add_argument("--days", type=int, default=7, help="Bugün dahil kaç gün taransın (1-14).")
    parser.add_argument("--workers", type=int, default=3, help="Eşzamanlı maç sayfası isteği (1-5).")
    parser.add_argument("--out", type=Path, default=Path("mackolik_upcoming.csv"))
    args = parser.parse_args()

    try:
        rows = collect_upcoming(days=args.days, workers=args.workers)
        write_csv(rows, args.out)
    except Exception as exc:
        print(f"HATA: Maçkolik verisi alınamadı: {exc}", file=sys.stderr)
        return 1

    print(f"Maçkolik'ten MS1/MSX/MS2 oranı alınan maç: {len(rows)}")
    print(f"Dosya: {args.out.resolve()}")
    if not rows:
        print("Seçilen gün aralığında oranı yayınlanmış yaklaşan maç bulunamadı.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
