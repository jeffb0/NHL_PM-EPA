import polars as pl

from src.calibration import team as tm


def test_goalie_regressor_is_joined_by_team_season(tmp_path, monkeypatch):
    # a two-team toy: skater sums by primary team plus the goalie value by defending team-season
    t = pl.DataFrame(
        {
            "season": [2020, 2020],
            "team": ["AAA", "BBB"],
            **{tm.RAW[c]: [1.0, 2.0] for c in tm.COMPS_ALL},
            **{tm.SHR[c]: [1.0, 2.0] for c in tm.COMPS_ALL},
        }
    )
    (tmp_path / "goalie_team_season.parquet").parent.mkdir(exist_ok=True)
    pl.DataFrame(
        {
            "season": [2020, 2020],
            "team": ["AAA", "BBB"],
            "gk_gar": [5.0, -3.0],
            "gk_raw_gar": [6.0, -2.0],
            "shots": [2000, 2100],
        }
    ).write_parquet(tmp_path / "goalie_team_season.parquet")
    monkeypatch.setattr(
        tm.recon,
        "team_gd_all",
        lambda proc: pl.DataFrame({"season": [2020, 2020], "team": ["AAA", "BBB"], "gd_all": [10, -10]}),
    )
    j = tm.team_table(t.group_by("season", "team").agg(pl.all().sum()), tmp_path, tm.COMPS_ALL_GK, target="gd_all")
    assert "gk_raw_gar" in j.columns and j.sort("team")["gk_gar"].to_list() == [5.0, -3.0] and j.height == 2
    assert tm.COMPS_ALL_GK[-1] == "gk" and tm.RAW["gk"] == "gk_raw_gar" and tm.SHR["gk"] == "gk_gar"
