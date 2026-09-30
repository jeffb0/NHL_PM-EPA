import polars as pl

from src.calibration import team as tm


def _players():
    return pl.DataFrame({"pos_group": ["F", "D"], "pm_gar": [2.0, 3.0], "def_gar": [1.0, 1.0], "fin_gar": [1.0, 1.0], "to_gar": [0.0, 0.0], "fo_gar": [0.5, 0.5], "pen_gar": [0.0, 0.0],
                         "st_pp_gar": [1.0, 1.0], "st_pk_gar": [4.0, 4.0], "st_4v4_gar": [0.0, 0.0], "st_3v3_gar": [0.0, 0.0], "total_gar_all": [9.5, 10.5]})


def test_position_weights_apply_by_position_and_ones_reproduce_the_total():
    w1 = {c: 1.0 for c in tm.COMPS_POS}
    same = tm.apply_weights_pos(_players(), {k: v for k, v in w1.items() if k != "gk"}, 5.9, "cal")
    assert same["cal_change_gar"].abs().max() < 1e-12
    w = dict(w1) | {"pk_F": 0.0, "pk_D": 2.0, "pm_F": 1.0, "pm_D": 2.0}
    x = tm.apply_weights_pos(_players(), {k: v for k, v in w.items() if k != "gk"}, 5.9, "pos")
    f, d = x.filter(pl.col("pos_group") == "F"), x.filter(pl.col("pos_group") == "D")
    assert abs(f["total_gar_all_pos"][0] - (9.5 - 4.0)) < 1e-12          # forwards' penalty kill switched off
    assert abs(d["total_gar_all_pos"][0] - (10.5 + 3.0 + 4.0)) < 1e-12   # defensemen's possession and penalty kill doubled


def test_position_split_team_table_sums_each_position_separately(tmp_path, monkeypatch):
    t = pl.DataFrame({"season": [2020, 2020], "team": ["AAA", "AAA"], "pos_group": ["F", "D"], **{tm.RAW[c]: [1.0, 10.0] for c in tm.SPLIT}, **{tm.SHR[c]: [1.0, 10.0] for c in tm.SPLIT},
                      **{tm.RAW[c]: [0.0, 0.0] for c in ("fo", "4v4", "3v3")}, **{tm.SHR[c]: [0.0, 0.0] for c in ("fo", "4v4", "3v3")}})
    pl.DataFrame({"season": [2020], "team": ["AAA"], "gk_gar": [1.0], "gk_raw_gar": [1.0]}).write_parquet(tmp_path / "goalie_team_season.parquet")
    monkeypatch.setattr(tm.recon, "team_gd_all", lambda proc: pl.DataFrame({"season": [2020], "team": ["AAA"], "gd_all": [5]}))
    j = tm.team_table(t, tmp_path, tm.COMPS_POS, target="gd_all")
    assert j[tm.RAW["pm_F"]][0] == 1.0 and j[tm.RAW["pm_D"]][0] == 10.0 and j[tm.SHR["pk_D"]][0] == 10.0
