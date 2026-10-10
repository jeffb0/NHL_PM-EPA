import json

from src.live.predict import overrides


def test_overrides_are_read_unless_switched_off(tmp_path):
    f = tmp_path / "o.json"
    o = {"unavailable": [{"player_id": 7}], "starters": {"AAA": 1}, "backups": {"AAA": 2}}
    f.write_text(json.dumps(o), encoding="utf-8")
    assert overrides(f) == ({7}, {"AAA": 1}, {"AAA": 2})
    f.write_text(json.dumps({**o, "active": False}), encoding="utf-8")
    assert overrides(f) == (set(), {}, {})                       # the lineup feed is then the only source of lineup news


def test_the_project_file_is_switched_off():
    assert overrides() == (set(), {}, {})
