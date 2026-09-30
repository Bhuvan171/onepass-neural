import numpy as np

from onepass_neural import georef


def rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_umeyama_recovers_known_similarity():
    src = np.random.default_rng(0).normal(size=(12, 3)) * 10
    dst = 2.5 * src @ rot_z(0.9).T + np.array([4.0, -3.0, 7.0])
    s, r, t = georef.umeyama(src, dst)
    assert abs(s - 2.5) < 1e-9
    assert np.allclose(r, rot_z(0.9), atol=1e-9)
    assert np.allclose(t, [4.0, -3.0, 7.0], atol=1e-9)
    assert np.linalg.det(r) > 0


def test_grid_trajectory_is_applied_and_straight_line_is_degenerate():
    # Three parallel flight lines (grid) vs a single straight line.
    xs, ys = np.meshgrid(np.arange(6) * 13.0, np.arange(3) * 30.0)
    enu = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 40.0)], 1)
    model = (enu - enu.mean(0)) @ rot_z(0.4) / 4.0  # model frame: rotated, 4x too small
    grid = georef.fit_georegistration(model, enu)
    assert grid["status"] == "applied"
    assert abs(grid["scale"] - 4.0) < 1e-6 and grid["residual_rmse_m"] < 1e-6

    line = georef.fit_georegistration(model[:6], enu[:6])
    assert line["status"] == "degenerate_near_collinear" and not line["applied"]
    assert abs(line["gps_scale_factor"] - 4.0) < 1e-6


def test_enu_axes():
    origin = (46.84, -91.99, 200.0)
    north = georef.geodetic_to_enu(np.array([46.84 + 1e-4]), np.array([-91.99]), np.array([200.0]), origin)[0]
    east = georef.geodetic_to_enu(np.array([46.84]), np.array([-91.99 + 1e-4]), np.array([200.0]), origin)[0]
    up = georef.geodetic_to_enu(np.array([46.84]), np.array([-91.99]), np.array([210.0]), origin)[0]
    assert 11.0 < north[1] < 11.2 and abs(north[0]) < 1e-3
    assert 7.5 < east[0] < 7.7 and abs(east[1]) < 1e-3
    assert abs(up[2] - 10.0) < 1e-3
