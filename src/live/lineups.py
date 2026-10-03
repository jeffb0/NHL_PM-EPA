"""Starting goalies and injury lists from the RotoWire NHL lineups page (docs/market/methodology.md §8).

Per game: each team's listed starter with its status (Confirmed / Expected) and the players listed as unavailable. Names are matched to NHL ids through the published rosters."""

from __future__ import annotations

import logging
import re
import unicodedata

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("live")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
URL = "https://www.rotowire.com/hockey/nhl-lineups.php"
ABBR = {"NJ": "NJD", "TB": "TBL", "LA": "LAK", "SJ": "SJS", "MON": "MTL", "WAS": "WSH", "CLB": "CBJ", "NAS": "NSH", "VEG": "VGK", "VGS": "VGK", "UTAH": "UTA", "CAL": "CGY", "WIN": "WPG", "ANH": "ANA"}
OUT_PREFIX = ("OUT", "IR", "SUSP", "LTIR")          # day-to-day and game-time decisions stay in the lineup; they are logged as questionable


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.replace("-", " "))).strip()


def _slug_name(a) -> str | None:
    """'/hockey/player/ukko-pekka-luukkonen-5516' -> 'ukko pekka luukkonen' (the link carries the full name even when the label is abbreviated)."""
    m = re.search(r"/player/([a-z0-9-]+?)-\d+$", a.get("href", "")) if a else None
    return norm(m.group(1)) if m else None


def parse(html: str) -> list[dict]:
    out = []
    for g in BeautifulSoup(html, "html.parser").find_all("div", class_="lineup"):
        teams = [ABBR.get(t.get_text(strip=True), t.get_text(strip=True)) for t in g.find_all(class_="lineup__abbr")]
        lists = g.find_all("ul", class_="lineup__list")
        if len(teams) != 2 or len(lists) != 2:
            continue
        game = {"away": teams[0], "home": teams[1], "teams": {}}
        for team, ul in zip(teams, lists):
            info = {"goalie": None, "goalie_status": None, "out": [], "questionable": []}
            section = None
            for li in ul.find_all("li"):
                cls = li.get("class") or []
                if "lineup__player-highlight" in cls:
                    info["goalie"] = _slug_name(li.find("a"))
                    txt = li.get_text(" ", strip=True)
                    info["goalie_status"] = "confirmed" if "Confirmed" in txt else "expected" if "Expected" in txt else "listed"
                elif "lineup__title" in cls:
                    section = li.get_text(" ", strip=True).upper()
                elif "lineup__player" in cls and section == "INJURIES":
                    name = _slug_name(li.find("a"))
                    inj = li.find(class_="lineup__inj")
                    status = (inj.get_text(strip=True) if inj else li.get_text(" ", strip=True).split()[-1]).upper()
                    if name:
                        (info["out"] if status.startswith(OUT_PREFIX) else info["questionable"]).append((name, status))
            game["teams"][team] = info
        out.append(game)
    return out


def fetch(tomorrow: bool = False, timeout: int = 30) -> list[dict]:
    try:
        r = requests.get(URL + ("?date=tomorrow" if tomorrow else ""), headers=UA, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("lineups fetch failed: %s", exc)
        return []
    return parse(r.text)


def resolve(game: dict, roster: dict[str, dict[str, int]]) -> dict:
    """Names -> player ids using ``roster[team] = {normalised full name: id}``; a last-name match is accepted when it is unique on the team. Unmatched names are returned for the log."""
    res = {}
    for team, info in game["teams"].items():
        names = roster.get(team, {})
        last: dict[str, list[int]] = {}
        for n, i in names.items():
            last.setdefault(n.split(" ")[-1], []).append(i)

        def pid(n: str | None, names=names, last=last) -> int | None:
            if not n:
                return None
            if n in names:
                return names[n]
            c = last.get(n.split(" ")[-1], [])
            return c[0] if len(c) == 1 else None

        out_ids, missed = set(), []
        for n, _ in info["out"]:
            i = pid(n)
            (out_ids.add(i) if i else missed.append(n))
        res[team] = {"goalie_id": pid(info["goalie"]), "goalie_name": info["goalie"], "goalie_status": info["goalie_status"], "out_ids": out_ids, "out_names": [n for n, _ in info["out"]],
                     "questionable": [n for n, _ in info["questionable"]], "unmatched": missed}
    return res
