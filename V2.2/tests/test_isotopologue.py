"""Isotopologue handling tests (run: python -m pytest V2.2/tests  or  python V2.2/tests/test_isotopologue.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce_lbl_twostream as m  # noqa: E402


def make_lines(species, isos, nu=1205.0):
    n = len(isos)
    f = lambda v: np.full(n, v, dtype=float)
    return m.HitranLines(species, m.MOLECULE_ID_BY_SPECIES[species], np.asarray(isos, dtype=np.int16),
                         f(nu), f(1e-21), f(1.0), f(0.07), f(0.2), f(1500.0), f(0.7), f(0.0))


def test_isotopologue_code_decoding():
    assert [m._isotopologue_id(c) for c in "1790AB"] == [1, 7, 9, 10, 11, 12]


def test_mass_per_isotopologue():
    lines = make_lines("H2O", [1, 4, 7])
    assert np.allclose(m.line_molecular_masses(lines), [18.010565, 19.01674, 20.022915])


def test_partition_ratio_is_isotopologue_specific():
    t = 1500.0
    main = float(m.log_partition_function_ratio("CH4", t, 1))
    ch3d = float(m.log_partition_function_ratio("CH4", t, 3))
    assert abs(main - ch3d) > 0.1          # ~25 % difference in Q-ratio
    assert abs(float(m.log_partition_function_ratio("CH4", 296.0, 3))) < 1e-9


def test_strength_uses_own_q():
    lines = make_lines("CH4", [1, 3])
    s = m.line_strength_at_temperature(lines, 1500.0)
    assert abs(s[0] / s[1] - 1.0) > 0.1    # identical inputs, different Q -> different S(T)


def test_isotope_scale_is_linear_in_strength():
    lines = make_lines("H2O", [1, 4])
    scaled = m.apply_isotopologue_scales(lines, m.parse_isotopologue_scales("H2O:4=3"))
    assert np.allclose(scaled.strength_ref, [1e-21, 3e-21])
    assert m.apply_isotopologue_scales(lines, {}) is lines


def test_bad_scale_inputs_rejected():
    for bad in ("H2O:9=2", "H2O4=2", "H2O:1=-1", "XX:1=1"):
        try:
            m.parse_isotopologue_scales(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_unknown_isotopologue_is_reported_early():
    try:
        m.validate_isotopologues(make_lines("CO", [1, 7]))
    except ValueError as exc:
        assert "7" in str(exc)
    else:
        raise AssertionError


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
