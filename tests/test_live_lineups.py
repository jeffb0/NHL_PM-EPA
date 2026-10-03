from src.live.lineups import norm, parse, resolve

HTML = """
<div class="lineup is-nhl"><div class="lineup__abbr">WPG</div><div class="lineup__abbr">TB</div>
<ul class="lineup__list is-visit">
<li class="lineup__player-highlight"><a href="/hockey/player/stuart-skinner-5559">S. Skinner</a> Expected</li>
<li class="lineup__title">POWER PLAY #1</li>
<li class="lineup__player">C <a href="/hockey/player/mark-scheifele-3782" title="Mark Scheifele">M. Scheifele</a></li>
<li class="lineup__title is-middle">INJURIES</li>
<li class="lineup__player">G <a href="/hockey/player/connor-hellebuyck-4567" title="Connor Hellebuyck">C. Hellebuyck</a> <span class="lineup__inj">SUSP</span></li>
<li class="lineup__player">LW <a href="/hockey/player/alex-iafallo-5000" title="Alex Iafallo">A. Iafallo</a> <span class="lineup__inj">DTD</span></li>
<li class="lineup__player">D <a href="/hockey/player/some-prospect-7000" title="Some Prospect">S. Prospect</a> <span class="lineup__inj">IR-NR</span></li>
</ul>
<ul class="lineup__list is-home">
<li class="lineup__player-highlight"><a href="/hockey/player/andrei-vasilevskiy-3900">A. Vasilevskiy</a> Confirmed</li>
<li class="lineup__title is-middle">INJURIES</li>
</ul></div>
"""


def test_parse_reads_goalies_injuries_and_maps_team_codes():
    g = parse(HTML)[0]
    assert (g["away"], g["home"]) == ("WPG", "TBL")
    w = g["teams"]["WPG"]
    assert w["goalie"] == "stuart skinner" and w["goalie_status"] == "expected"
    assert [n for n, _ in w["out"]] == ["connor hellebuyck", "some prospect"]          # SUSP and IR-NR are out; the power-play list is not an injury list
    assert [n for n, _ in w["questionable"]] == ["alex iafallo"]                        # day-to-day stays in the lineup
    assert g["teams"]["TBL"]["goalie_status"] == "confirmed" and g["teams"]["TBL"]["out"] == []


def test_resolve_matches_names_and_reports_the_unmatched():
    roster = {"WPG": {norm("Stuart Skinner"): 1, norm("Connor Hellebuyck"): 2, norm("Alex Iafallo"): 3}, "TBL": {norm("Andrei Vasilevskiy"): 9}}
    r = resolve(parse(HTML)[0], roster)
    assert r["WPG"]["goalie_id"] == 1 and r["WPG"]["out_ids"] == {2} and r["WPG"]["unmatched"] == ["some prospect"]
    assert r["TBL"]["goalie_id"] == 9 and r["TBL"]["out_ids"] == set()
    assert norm("Ukko-Pekka Luukkonen") == "ukko pekka luukkonen" and norm("Tim Stützle") == "tim stutzle"
