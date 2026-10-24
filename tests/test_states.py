"""State mapping: exhaustive + mutually exclusive, correct geometry, correct kinds."""

import numpy as np
import polars as pl

from src.states.mapping import add_state_columns, coarse_of, grid_of, location_precision, state_kind, zone_of


def test_zone_geometry_landmarks():
    pts = {
        (80, 0): "OZ_low_slot",
        (60, 5): "OZ_high_slot",
        (95, 3): "OZ_behind",
        (75, 20): "OZ_circle_L",
        (75, -20): "OZ_circle_R",
        (80, 38): "OZ_boards_L",
        (40, 0): "OZ_point_C",
        (40, -30): "OZ_point_R",
        (10, 0): "NZ_att_C",
        (-10, 30): "NZ_own_L",
        (-80, 0): "DZ_low_slot",
        (-95, 0): "DZ_behind",
        (-60, -38): "DZ_boards_R",
    }
    xs, ys = zip(*pts)
    got = zone_of(np.array(xs, float), np.array(ys, float))
    assert list(got) == list(pts.values())


def test_zones_are_exhaustive_and_exactly_26():
    xs, ys = np.meshgrid(np.linspace(-99.5, 99.5, 400), np.linspace(-42, 42, 170))
    z = zone_of(xs.ravel(), ys.ravel())
    assert "NA" not in set(z)
    assert "" not in {s.split("_", 1)[1] for s in z}
    assert len(set(z)) == 26


def test_blue_line_boundaries():
    assert list(coarse_of(np.array([25.0, 24.9, -25.0, -24.9]))) == ["OZ", "NZ", "DZ", "NZ"]


def test_grid_cells_cover_rink_and_clip_edges():
    g = grid_of(np.array([-100.0, 99.99, 100.0]), np.array([-42.5, 42.49, 42.5]), 10, 8.5)
    assert list(g) == ["0_0", "19_9", "19_9"]


def test_missing_coordinates_map_to_na():
    assert zone_of(np.array([np.nan]), np.array([1.0]))[0] == "NA"


def test_kinds_and_precision():
    kinds = state_kind(
        np.array(
            ["takeaway", "shot", "faceoff", "loose_opp_miss", "loose_opp_giveaway", "loose_unobserved"], dtype=object
        )
    )
    assert list(kinds) == ["control", "control", "faceoff", "loose_opp_miss", "loose_opp_giveaway", "loose_unobserved"]
    assert list(location_precision(kinds)) == ["fine", "fine", "fine", "coarse", "fine", "none"]


def test_state_ids_use_strength_group_and_coarse_for_imprecise_loose():
    df = pl.DataFrame(
        {
            "obs_kind": ["takeaway", "loose_opp_miss", "faceoff"],
            "x": [80.0, -70.0, 0.0],
            "y": [0.0, 10.0, 0.0],
            "strength": ["5v5", "3v3", "PP"],
            "score_diff": [0, 1, -3],
        }
    )
    out = add_state_columns(df, "zones")
    assert out["state_id"].to_list() == ["5v5|control|OZ_low_slot", "4v4_3v3|loose_opp_miss|DZ", "PP|faceoff|NZ_att_C"]
    out = add_state_columns(df, "zones", use_score=True)
    assert out["state_id"][2].endswith("|s-2")  # score clipped at -2
