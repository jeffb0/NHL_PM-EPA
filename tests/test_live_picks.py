from src.live.picks import decimal, fmt, payout, units


def test_payout():
    assert abs(payout(-125) - 0.8) < 1e-12 and abs(payout(230) - 2.3) < 1e-12
    assert abs(decimal(-125) - 1.8) < 1e-12 and fmt(230) == "3.30" and fmt(-275) == "1.36" and fmt(100) == "2.00"


def test_units_scale_with_the_gap_and_are_capped():
    # model less than 3 points above break-even (52.4%): no pick
    assert units(0.52, 0.50, -110)[0] == 0
    assert units(0.56, 0.50, -110)[0] == 1  # modest gap: one unit
    assert units(0.62, 0.50, -110)[0] == 2
    assert units(0.80, 0.50, -110)[0] == 3  # capped at three units however large the gap
    u, p = units(0.56, 0.50, -110)
    assert abs(p - 0.53) < 1e-12  # sizing probability is halfway between market and model


def test_no_pick_when_the_halved_gap_does_not_beat_the_price():
    # model 3.5 points over break-even at -200 (66.7%), but the market is at 62%: halfway (66.1%) is below break-even,
    # so the Kelly fraction is negative
    assert units(0.702, 0.62, -200)[0] == 0
    # for the same gap over break-even (10 points model, 5 after halving) an underdog gets fewer units than a near-even
    # side
    assert units(0.403, 0.303, 230)[0] == 2 and units(0.6238, 0.5238, -110)[0] == 3
