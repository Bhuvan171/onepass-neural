"""Post-hoc georegistration: fit a similarity transform from predicted camera
centres to GPS positions (local ENU metres).

This is NOT GNSS-constrained bundle adjustment. GPS residuals measure agreement
with the (noisy) input GPS, not ground accuracy.
"""

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def read_exif_gps(path):
    """Return (lat, lon, alt) in degrees/metres, or None if the image has no GPS."""
    import exifread

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if "GPS GPSLatitude" not in tags or "GPS GPSLongitude" not in tags:
        return None

    def dms(tag):
        d, m, s = [float(x.num) / float(x.den) for x in tag.values]
        return d + m / 60 + s / 3600

    lat = dms(tags["GPS GPSLatitude"]) * (1 if str(tags.get("GPS GPSLatitudeRef", "N")) == "N" else -1)
    lon = dms(tags["GPS GPSLongitude"]) * (1 if str(tags.get("GPS GPSLongitudeRef", "E")) == "E" else -1)
    alt = None
    if "GPS GPSAltitude" in tags:
        r = tags["GPS GPSAltitude"].values[0]
        alt = float(r.num) / float(r.den)
    return lat, lon, alt


def geodetic_to_ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], -1)


def geodetic_to_enu(lat, lon, alt, origin):
    """ENU (x east, y north, z up) metres relative to origin=(lat0, lon0, alt0)."""
    lat0, lon0, alt0 = origin
    d = geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(lat0, lon0, alt0)
    la, lo = np.radians(lat0), np.radians(lon0)
    r = np.array([
        [-np.sin(lo), np.cos(lo), 0],
        [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
    ])
    return d @ r.T


def umeyama(src, dst):
    """Least-squares similarity dst ~ s * R @ src + t (Umeyama 1991), no reflections."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(d) @ sign) / var_s
    t = mu_d - s * r @ mu_s
    return s, r, t


def trajectory_shape(enu):
    """Singular values of centred GPS positions. s2/s1 small => near-collinear."""
    sv = np.linalg.svd(enu - enu.mean(0), compute_uv=False)
    return sv, float(sv[1] / sv[0]) if sv[0] > 0 else 0.0


def fit_georegistration(cam_centres, enu, inlier_m=5.0, collinear_ratio=0.1, iters=500, seed=0):
    """Robust similarity fit from model camera centres to GPS ENU positions.

    Returns a dict describing the result. If the GPS trajectory is near-collinear,
    rotation about the flight line is unconstrained: only the scale is reported
    and no geographic placement is applied.
    """
    n = len(enu)
    sv, ratio = trajectory_shape(enu)
    out = {
        "method": "post-hoc similarity georegistration (Umeyama + RANSAC) of predicted camera centres to GPS",
        "num_cameras_with_gps": int(n),
        "gps_trajectory_singular_values_m": [round(float(x), 3) for x in sv],
        "gps_trajectory_s2_over_s1": round(ratio, 4),
        "validation_status": "unvalidated",
        "note": "Residuals measure agreement with input GPS, not ground-truth accuracy.",
    }
    if n < 3:
        out.update(status="insufficient_gps", applied=False)
        return out

    # Scale is recoverable even for a straight line: ratio of spreads.
    spread_scale = float(np.sqrt(((enu - enu.mean(0)) ** 2).sum() / ((cam_centres - cam_centres.mean(0)) ** 2).sum()))

    if ratio < collinear_ratio:
        out.update(
            status="degenerate_near_collinear",
            applied=False,
            gps_scale_factor=round(spread_scale, 4),
            reason="GPS positions lie nearly on a line; rotation about the flight line is not "
                   "determined, so no geographic placement is applied. Only a GPS-derived scale is reported.",
        )
        return out

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if trajectory_shape(enu[idx])[1] < collinear_ratio:
            continue
        s, r, t = umeyama(cam_centres[idx], enu[idx])
        err = np.linalg.norm((s * cam_centres @ r.T + t) - enu, axis=1)
        inl = err < inlier_m
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        best = np.ones(n, bool)
    s, r, t = umeyama(cam_centres[best], enu[best])
    resid = np.linalg.norm((s * cam_centres @ r.T + t) - enu, axis=1)

    # Leave-one-out: fit without camera i, measure its GPS disagreement.
    loo = []
    for i in np.where(best)[0]:
        keep = best.copy()
        keep[i] = False
        if keep.sum() >= 3 and trajectory_shape(enu[keep])[1] >= collinear_ratio:
            si, ri, ti = umeyama(cam_centres[keep], enu[keep])
            loo.append(float(np.linalg.norm(si * ri @ cam_centres[i] + ti - enu[i])))

    out.update(
        status="applied",
        applied=True,
        scale=float(s),
        rotation=r.tolist(),
        translation=t.tolist(),
        inliers=int(best.sum()),
        inlier_threshold_m=inlier_m,
        residual_rmse_m=round(float(np.sqrt((resid[best] ** 2).mean())), 3),
        residual_max_m=round(float(resid[best].max()), 3),
        residuals_m=[round(float(x), 3) for x in resid],
        leave_one_out_rmse_m=round(float(np.sqrt(np.mean(np.square(loo)))), 3) if loo else None,
        leave_one_out_max_m=round(float(max(loo)), 3) if loo else None,
    )
    return out
