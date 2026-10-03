"""Pre-game prices from the scoresandodds.com comparison page (docs/market/methodology.md §2)."""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("live")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
SLUG = {"ANA": "ducks", "BOS": "bruins", "BUF": "sabres", "CAR": "hurricanes", "CBJ": "blue-jackets", "CGY": "flames", "CHI": "blackhawks", "COL": "avalanche", "DAL": "stars", "DET": "red-wings", "EDM": "oilers",
        "FLA": "panthers", "LAK": "kings", "MIN": "wild", "MTL": "canadiens", "NJD": "devils", "NSH": "predators", "NYI": "islanders", "NYR": "rangers", "OTT": "senators", "PHI": "flyers", "PIT": "penguins",
        "SEA": "kraken", "SJS": "sharks", "STL": "blues", "TBL": "lightning", "TOR": "maple-leafs", "UTA": "mammoth", "VAN": "canucks", "VGK": "golden-knights", "WPG": "jets", "WSH": "capitals"}
BOOK = {"fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM", "bet365": "bet365", "caesars": "Caesars", "fanatics": "Fanatics", "riverscasino": "BetRivers"}


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
    """Rows of {book, market, side, line, american} from a game page; side is 'away'/'home' for the moneyline and 'over'/'under' for the total."""
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
                val, odds, ml = td.find(class_="data-value"), td.find(class_="data-odds"), td.find(class_="data-moneyline")
                v, o = (val.get_text(strip=True) if val else ""), (odds.get_text(strip=True) if odds else "")
                price = american(ml.get_text(strip=True)) if ml else american(o) if o else None
                if price is None:
                    continue
                line = float(v[1:]) if market == "total" and re.fullmatch(r"[ou]\d+(\.\d)?", v) else None
                out.append({"book": BOOK.get(b, b), "market": market, "side": side, "line": line, "american": price})
    return out


def fetch(away: str, home: str, timeout: int = 30) -> tuple[list[dict], str]:
    """Prices for a matchup. The page is keyed by the two nicknames in alphabetical order and shows the teams' next meeting."""
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
        log.warning("odds page for %s at %s shows a different game (%s)", away, home, title)        # e.g. the reverse fixture: prices would be attached to the wrong sides
        return [], url
    return parse(r.text), url
