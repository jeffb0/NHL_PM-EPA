"""Pre-game prices from the scoresandodds.com comparison page (docs/market/methodology.md §2)."""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("live")
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
}
SLUG = {
    "ANA": "ducks",
    "BOS": "bruins",
    "BUF": "sabres",
    "CAR": "hurricanes",
    "CBJ": "blue-jackets",
    "CGY": "flames",
    "CHI": "blackhawks",
    "COL": "avalanche",
    "DAL": "stars",
    "DET": "red-wings",
    "EDM": "oilers",
    "FLA": "panthers",
    "LAK": "kings",
    "MIN": "wild",
    "MTL": "canadiens",
    "NJD": "devils",
    "NSH": "predators",
    "NYI": "islanders",
    "NYR": "rangers",
    "OTT": "senators",
    "PHI": "flyers",
    "PIT": "penguins",
    "SEA": "kraken",
    "SJS": "sharks",
    "STL": "blues",
    "TBL": "lightning",
    "TOR": "maple-leafs",
    "UTA": "mammoth",
    "VAN": "canucks",
    "VGK": "golden-knights",
    "WPG": "jets",
    "WSH": "capitals",
}
BOOK = {
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "bet365": "bet365",
    "caesars": "Caesars",
    "fanatics": "Fanatics",
    "riverscasino": "BetRivers",
}


def american(text: str) -> int | None:
    """'-105', '+210', 'even' -> American odds as an integer."""
    t = text.strip().lower()
    if t in ("even", "ev"):
        return 100
    m = re.fullmatch(r"[+-]\d{3,5}", t)
    return int(t) if m else None


def implied(a: int) -> float:
    return 100.0 / (a + 100.0) if a > 0 else -a / (-a + 100.0)


def devig(away: int, home: int) -> tuple[float, float]:
    """Two-way prices to probabilities that sum to one (margin removed proportionally)."""
    pa, ph = implied(away), implied(home)
    return pa / (pa + ph), ph / (pa + ph)


def parse(html: str) -> list[dict]:
    """Rows of {book, market, side, line, american} from a game page; side is 'away'/'home' for the moneyline and
    'over'/'under' for the total."""
    t = BeautifulSoup(html, "html.parser").find("table", class_="odds-table")
    if t is None:
        return []
    books = [i.get("alt", "") for i in t.find("thead").find_all("img")]
    out = []
    for tb in t.find_all("tbody"):
        cls = " ".join(tb.get("class") or [])
        market = "moneyline" if "moneyline" in cls else "total" if "total" in cls else None
        rows = tb.find_all("tr")
        if market is None or len(rows) != 2:
            continue
        for side, tr in zip(("away", "home") if market == "moneyline" else ("over", "under"), rows):
            cells = tr.find_all("td", class_="game-odds")
            if len(cells) != len(books):
                continue
            for b, td in zip(books, cells):
                val, odds, ml = (
                    td.find(class_="data-value"),
                    td.find(class_="data-odds"),
                    td.find(class_="data-moneyline"),
                )
                v, o = (val.get_text(strip=True) if val else ""), (odds.get_text(strip=True) if odds else "")
                price = american(ml.get_text(strip=True)) if ml else american(o) if o else None
                if price is None:
                    continue
                line = float(v[1:]) if market == "total" and re.fullmatch(r"[ou]\d+(\.\d)?", v) else None
                out.append({"book": BOOK.get(b, b), "market": market, "side": side, "line": line, "american": price})
    return out


# books the comparison page does not carry, from Action Network's scoreboard feed (New Jersey lines;
# docs/market/methodology.md §8 item 13)
AN_URL = "https://api.actionnetwork.com/web/v2/scoreboard/nhl"
AN_BOOKS = {2724: "Hard Rock Bet", 4620: "theScore Bet"}
AN_ABBR = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}


def parse_an(d: dict) -> dict[tuple[str, str], list[dict]]:
    """Moneyline and total prices per (away, home) from an Action Network scoreboard response, for the books in
    AN_BOOKS."""
    out: dict = {}
    for g in d.get("games", []):
        ab = {t["id"]: AN_ABBR.get(t["abbr"], t["abbr"]) for t in g.get("teams", [])}
        side_of = {g["away_team_id"]: "away", g["home_team_id"]: "home"}
        rows = []
        for bid, m in (g.get("markets") or {}).items():
            book = AN_BOOKS.get(int(bid))
            ev = (m or {}).get("event") or {}
            if book is None:
                continue
            for o in ev.get("moneyline") or []:
                if o.get("team_id") in side_of and isinstance(o.get("odds"), int) and not o.get("is_live"):
                    rows.append(
                        {
                            "book": book,
                            "market": "moneyline",
                            "side": side_of[o["team_id"]],
                            "line": None,
                            "american": o["odds"],
                        }
                    )
            for o in ev.get("total") or []:
                if o.get("side") in ("over", "under") and isinstance(o.get("odds"), int) and not o.get("is_live"):
                    rows.append(
                        {
                            "book": book,
                            "market": "total",
                            "side": o["side"],
                            "line": float(o["value"]) if o.get("value") is not None else None,
                            "american": o["odds"],
                        }
                    )
        if g.get("away_team_id") in ab and g.get("home_team_id") in ab:
            out[(ab[g["away_team_id"]], ab[g["home_team_id"]])] = rows
    return out


def fetch_extra(day: str, timeout: int = 30) -> tuple[dict[tuple[str, str], list[dict]], str]:
    """Hard Rock Bet and theScore Bet prices for every game on ``day`` (an Eastern date, YYYY-MM-DD), keyed by (away,
    home). Empty on failure: the other books are still logged."""
    url = f"{AN_URL}?bookIds={','.join(map(str, AN_BOOKS))}&periods=event&date={day.replace('-', '')}"
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        return parse_an(r.json()), url
    except (requests.RequestException, ValueError) as exc:
        log.warning("Action Network fetch failed for %s: %s", day, exc)
        return {}, url


def fetch(away: str, home: str, timeout: int = 30) -> tuple[list[dict], str]:
    """Prices for a matchup. The page is keyed by the two nicknames in alphabetical order and shows the teams' next
    meeting."""
    a, h = sorted([SLUG[away], SLUG[home]])
    url = f"https://www.scoresandodds.com/nhl/{a}-vs-{h}"
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("odds fetch failed for %s at %s: %s", away, home, exc)
        return [], url
    head = BeautifulSoup(r.text, "html.parser").find("h1")
    title = head.get_text(" ", strip=True).lower() if head else ""
    want = f"{SLUG[away].replace('-', ' ')} @ {SLUG[home].replace('-', ' ')}"
    if want.split(" @ ")[0] not in title.split("@")[0] or want.split(" @ ")[1] not in title.split("@")[-1]:
        # e.g. the reverse fixture: prices would be attached to the wrong sides
        log.warning("odds page for %s at %s shows a different game (%s)", away, home, title)
        return [], url
    return parse(r.text), url
