"""GPU integration test: real images -> real artifact. Takes ~30 s."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

from onepass_neural import pipeline

ROOT = Path(__file__).resolve().parent.parent
IMAGES = sorted((ROOT / "data" / "brighton_beach" / "images").glob("*.JPG"))


def make_job(tmp_path, images):
    (tmp_path / "job.json").write_text(json.dumps({"images": [str(p) for p in images], "label": "test"}))
    return tmp_path


def test_images_to_artifact(tmp_path):
    job = make_job(tmp_path, IMAGES[:4])
    pipeline.main(job)
    v = PlyData.read(str(job / "cloud.ply"))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    assert len(xyz) > 10000 and np.isfinite(xyz).all()
    manifest = json.loads((job / "artifact_manifest.json").read_text())
    assert manifest["artifacts"]["cloud.ply"]["sha256"] == hashlib.sha256((job / "cloud.ply").read_bytes()).hexdigest()
    metrics = json.loads((job / "metrics.json").read_text())
    assert metrics["points_exported"] == len(xyz)
    assert metrics["validation_status"] == "unvalidated"
    # 4 images from one flight line: GPS is collinear, so no geolocation may be claimed.
    assert metrics["georef_status"] == "degenerate_near_collinear" and metrics["geolocation"] == "none"


def test_single_image_fails_without_artifact(tmp_path):
    job = make_job(tmp_path, IMAGES[:1])
    with pytest.raises(SystemExit):
        pipeline.main(job)
    assert not (job / "cloud.ply").exists()
