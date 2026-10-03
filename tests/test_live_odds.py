from src.live.odds import american, devig, implied, parse

HTML = """
<table class="odds-table game-table"><thead><tr><th class="game-team" colspan="2">x</th>
<th class="book-logo"><img alt="betmgm"/></th><th class="book-logo"><img alt="draftkings"/></th></tr></thead>
<tbody class="odds-table-spread--1"><tr><td class="game-time"></td><td class="game-team">A</td>
<td class="game-odds"><span class="data-value">+1.5</span><small class="data-odds">-260</small></td><td class="game-odds"><span class="data-value">+1.5</span><small class="data-odds">-258</small></td></tr>
<tr><td class="game-team">B</td><td class="game-odds"><span class="data-value">-1.5</span><small class="data-odds">+210</small></td><td class="game-odds"><span class="data-value">-1.5</span><small class="data-odds">+210</small></td></tr></tbody>
<tbody class="odds-table-total--1"><tr><td class="game-time"></td><td class="game-team">A</td>
<td class="game-odds"><span class="data-value">o5.5</span><small class="data-odds">-125</small></td><td class="game-odds"><span class="data-value">o5.5</span><small class="data-odds">-125</small></td></tr>
<tr><td class="game-team">B</td><td class="game-odds"><span class="data-value">u5.5</span><small class="data-odds">+105</small></td><td class="game-odds"><span class="data-value">u5.5</span><small class="data-odds">even</small></td></tr></tbody>
<tbody class="odds-table-moneyline--1 active"><tr><td class="game-time"></td><td class="game-team">A</td>
<td class="game-odds"><span class="data-moneyline">-110</span></td><td class="game-odds"><span class="data-moneyline">-105</span></td></tr>
<tr><td class="game-team">B</td><td class="game-odds"><span class="data-moneyline">-110</span></td><td class="game-odds"></td></tr></tbody></table>
"""


def test_american_and_devig():
    assert american("-105") == -105 and american("+210") == 210 and american("even") == 100 and american("o5.5") is None
    assert abs(implied(-110) - 110 / 210) < 1e-12 and abs(implied(150) - 0.4) < 1e-12
    a, h = devig(-105, -115)
    assert abs(a + h - 1) < 1e-12 and abs(a - 0.4891) < 1e-3


def test_parse_keeps_books_sides_and_skips_missing_prices():
    rows = parse(HTML)
    ml = {(r["book"], r["side"]): r["american"] for r in rows if r["market"] == "moneyline"}
    assert ml == {("BetMGM", "away"): -110, ("DraftKings", "away"): -105, ("BetMGM", "home"): -110}          # DraftKings has no home price: skipped, not guessed
    tot = {(r["book"], r["side"]): (r["line"], r["american"]) for r in rows if r["market"] == "total"}
    assert tot[("DraftKings", "under")] == (5.5, 100) and tot[("BetMGM", "over")] == (5.5, -125)
    assert not [r for r in rows if r["market"] not in ("moneyline", "total")]                                  # the puck line is not logged
