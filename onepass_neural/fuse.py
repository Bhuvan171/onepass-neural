"""Per-view MapAnything outputs -> one coloured point cloud (PLY)."""

import numpy as np
from plyfile import PlyData, PlyElement


def collect_points(views):
    """Concatenate all valid (masked) pixels of all views."""
    pts, rgb, mvc, conf, view_id = [], [], [], [], []
    for i, v in enumerate(views):
        m = v["mask"] & np.isfinite(v["pts3d"]).all(-1)
        pts.append(v["pts3d"][m])
        rgb.append(v["rgb"][m])
        mvc.append(v["mv_consistency"][m])
        conf.append(v["conf"][m])
        view_id.append(np.full(m.sum(), i, np.uint16))
    return {
        "xyz": np.concatenate(pts).astype(np.float64),
        "rgb": np.concatenate(rgb),
        "mv_consistency": np.concatenate(mvc).astype(np.float32),
        "model_conf": np.concatenate(conf).astype(np.float32),
        "view_id": np.concatenate(view_id),
    }


def voxel_downsample(cloud, voxel_size):
    """Keep one point per voxel (the first one encountered). Simple and deterministic."""
    keys = np.floor(cloud["xyz"] / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return {k: v[keep] for k, v in cloud.items()}


def auto_voxel_size(xyz, points_across=1500):
    """Voxel size so the larger horizontal extent spans ~points_across voxels."""
    span = np.percentile(xyz, 98, axis=0) - np.percentile(xyz, 2, axis=0)
    return float(max(span[0], span[1]) / points_across)


def write_ply(path, cloud):
    n = len(cloud["xyz"])
    data = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("mv_consistency", "f4"), ("model_conf", "f4"), ("view_id", "u2"),
    ])
    data["x"], data["y"], data["z"] = cloud["xyz"].T.astype(np.float32)
    data["red"], data["green"], data["blue"] = cloud["rgb"].T
    data["mv_consistency"] = cloud["mv_consistency"]
    data["model_conf"] = cloud["model_conf"]
    data["view_id"] = cloud["view_id"]
    PlyData([PlyElement.describe(data, "vertex")], byte_order="<").write(str(path))
