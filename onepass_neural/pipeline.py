"""One reconstruction run: job.json -> selected frames -> MapAnything -> GPS
alignment -> cloud.ply + manifests, all inside the job directory.

Usage: python -m onepass_neural.pipeline <job_dir>
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from onepass_neural import fuse, georef, selection
from onepass_neural.infer import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parent.parent
MAPANYTHING_DIR = ROOT / "third_party" / "map-anything"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


class Events:
    def __init__(self, path):
        self.path, self.t0, self.stages = path, time.perf_counter(), {}

    def log(self, **kw):
        kw["t"] = round(time.perf_counter() - self.t0, 3)
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    def start(self, stage):
        self.stages[stage] = time.perf_counter()
        self.log(stage=stage, event="start")

    def end(self, stage, **kw):
        secs = round(time.perf_counter() - self.stages[stage], 3)
        self.stages[stage] = secs
        self.log(stage=stage, event="end", seconds=secs, **kw)


def environment():
    import torch
    commit = subprocess.run(["git", "-C", str(MAPANYTHING_DIR), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {
        "hostname": platform.node(), "os": platform.platform(), "python": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(), "gpu": gpu,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "mapanything_commit": commit, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0 (Hugging Face model card)",
    }


def gather_frames(job, job_dir, ev):
    """Return (frames, input_manifest). Each frame: path, source, gps (or None)."""
    if job.get("video"):
        video = Path(job["video"])
        info = selection.probe_video(video)
        ev.log(stage="ingest", event="video_probe", **info)
        frames = selection.extract_video_frames(video, job_dir / "frames", job.get("sample_fps", 2.0))
        srt = selection.parse_dji_srt(job["srt"]) if job.get("srt") else []
        for f in frames:
            r = selection.gps_for_timestamp(srt, f["timestamp_s"]) if srt else None
            f["gps"] = (r["lat"], r["lon"], r["alt"]) if r else None
            f["gps_alt_type"] = r["alt_type"] if r else None
            f["source"] = f"{video.name}@{f['timestamp_s']}s"
        manifest = {
            "input_type": "video", "video": str(video), "video_sha256": sha256(video), "video_info": info,
            "telemetry": {"srt": str(job["srt"]), "srt_sha256": sha256(job["srt"]), "records_with_fix": len(srt)}
            if job.get("srt") else None,
            "frame_sampling": f"decoded at {job.get('sample_fps', 2.0)} fps; timestamp = index / fps",
        }
    else:
        frames = []
        for p in job["images"]:
            g = georef.read_exif_gps(p)
            frames.append({"path": Path(p), "source": Path(p).name, "gps": g,
                           "gps_alt_type": "EXIF GPSAltitude (datum unknown)" if g else None})
        manifest = {
            "input_type": "image_set",
            "images": [{"name": Path(p).name, "sha256": sha256(p)} for p in job["images"]],
            "telemetry": "EXIF GPS in images" if any(f["gps"] for f in frames) else None,
        }
    manifest["provenance"] = job.get("provenance", "user upload")
    manifest["label"] = job.get("label")
    return frames, manifest


def main(job_dir):
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    ev = Events(job_dir / "events.jsonl")
    write_json(job_dir / "environment.json", environment())
    config = {"max_views": job.get("max_views", 40), "voxel_points_across": 600,
              "georef_inlier_threshold_m": 5.0, "selector": "quality-filtered interval sampling (not parallax-aware)",
              "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "resolution_set": 518,
                        "amp": "bf16", "memory_efficient_inference": True, "mask_edges": True}}
    write_json(job_dir / "config.json", config)
    t_all = time.perf_counter()

    # 1. ingest + selection
    ev.start("ingest_and_select")
    frames, input_manifest = gather_frames(job, job_dir, ev)
    frames = selection.quality_filter(frames)
    frames = selection.interval_sample(frames, config["max_views"])
    accepted = [f for f in frames if f["accepted"]]
    write_json(job_dir / "input_manifest.json", input_manifest)
    thumbs = job_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    import cv2
    with open(job_dir / "frames.jsonl", "w") as fh:
        for i, f in enumerate(frames):
            img = cv2.imread(str(f["path"]))
            if img is not None:
                cv2.imwrite(str(thumbs / f"{i:05d}.jpg"), cv2.resize(img, (240, int(240 * img.shape[0] / img.shape[1]))))
            fh.write(json.dumps({
                "index": i, "source": f["source"], "timestamp_s": f.get("timestamp_s"),
                "accepted": f["accepted"], "reason": f["reason"], "quality": f["quality"],
                "gps": f["gps"], "gps_alt_type": f["gps_alt_type"], "thumb": f"thumbs/{i:05d}.jpg",
            }) + "\n")
    ev.end("ingest_and_select", candidates=len(frames), accepted=len(accepted))
    if len(accepted) < 2:
        raise SystemExit(f"Only {len(accepted)} usable frame(s); need at least 2 overlapping views.")

    # 2. reconstruction
    ev.start("model_load")
    from onepass_neural.infer import load_model, run_inference
    model = load_model()
    ev.end("model_load")
    ev.start("reconstruct")
    views, infer_stats = run_inference(model, [f["path"] for f in accepted])
    ev.end("reconstruct", **infer_stats)
    del model

    # 3. GPS alignment
    ev.start("georegister")
    cams = np.array([v["cam2world"] for v in views], dtype=np.float64)  # cam2world, OpenCV axes
    centres = cams[:, :3, 3]  # cam2world translation is the camera centre
    gps_idx = [i for i, f in enumerate(accepted) if f["gps"] and f["gps"][2] is not None]
    cloud = fuse.collect_points(views)
    geo = {"status": "no_gps", "applied": False, "reason": "no GPS available for the selected frames"}
    frame_name = "model frame: first camera, OpenCV axes (x right, y down, z forward), learned metric scale"
    scale_source, geolocation = "learned_prior", "none"
    if len(gps_idx) >= 3:
        gps = np.array([accepted[i]["gps"] for i in gps_idx])
        origin = tuple(gps[0])
        enu = georef.geodetic_to_enu(gps[:, 0], gps[:, 1], gps[:, 2], origin)
        geo = georef.fit_georegistration(centres[gps_idx], enu, inlier_m=config["georef_inlier_threshold_m"])
        geo["enu_origin_lat_lon_alt"] = list(origin)
        geo["altitude_type"] = accepted[gps_idx[0]]["gps_alt_type"]
        geo["learned_to_gps_scale_ratio"] = geo.get("scale", geo.get("gps_scale_factor"))
        if geo["applied"]:
            s, r, t = geo["scale"], np.array(geo["rotation"]), np.array(geo["translation"])
            cloud["xyz"] = s * cloud["xyz"] @ r.T + t
            centres = s * centres @ r.T + t
            cams[:, :3, :3] = r @ cams[:, :3, :3]
            frame_name = "local ENU metres (x east, y north, z up) about enu_origin_lat_lon_alt; vertical datum unknown"
            scale_source, geolocation = "sensor_constrained (post-hoc GPS similarity)", "post-hoc georegistration (unvalidated)"
        elif "gps_scale_factor" in geo:
            s = geo["gps_scale_factor"]
            cloud["xyz"] *= s
            centres = centres * s
            frame_name = "model frame (first camera, OpenCV axes) rescaled by GPS spread ratio; not geolocated"
            scale_source = "gps_spread_ratio (degenerate trajectory, rotation unconstrained)"
    write_json(job_dir / "georef.json", geo)
    ev.end("georegister", status=geo["status"])

    # 4. fuse + export
    ev.start("fuse_export")
    n_raw = len(cloud["xyz"])
    voxel = fuse.auto_voxel_size(cloud["xyz"], config["voxel_points_across"])
    cloud = fuse.voxel_downsample(cloud, voxel)
    if not np.isfinite(cloud["xyz"]).all() or len(cloud["xyz"]) == 0:
        raise SystemExit("Reconstruction produced empty or non-finite geometry.")
    fuse.write_ply(job_dir / "cloud.ply", cloud)
    write_json(job_dir / "cameras.json", {
        "frame": frame_name,
        "cameras": [{"source": accepted[i]["source"], "centre": centres[i].tolist(),
                     "forward": cams[i, :3, 2].tolist(), "up_in_image": (-cams[i, :3, 1]).tolist()}
                    for i in range(len(views))],
    })
    ev.end("fuse_export", points_raw=n_raw, points_exported=len(cloud["xyz"]), voxel_size=voxel)

    mvc = cloud["mv_consistency"]
    metrics = {
        "backend": f"MapAnything feed-forward reconstruction ({MODEL_ID}@{MODEL_REVISION[:8]})",
        "result_type": "AI preview — spatial accuracy not validated",
        "input_label": job.get("label"), "input_type": input_manifest["input_type"],
        "candidate_frames": len(frames), "accepted_frames": len(accepted),
        "frames_with_gps": len(gps_idx), "points_before_downsample": n_raw,
        "points_exported": int(len(cloud["xyz"])), "voxel_size": round(voxel, 4),
        "coordinate_frame": frame_name, "scale_source": scale_source, "geolocation": geolocation,
        "validation_status": "unvalidated",
        "georef_status": geo["status"],
        "georef_rmse_m": geo.get("residual_rmse_m"), "georef_loo_rmse_m": geo.get("leave_one_out_rmse_m"),
        "learned_to_gps_scale_ratio": geo.get("learned_to_gps_scale_ratio"),
        "mv_consistency_ge_0_5_fraction": round(float((mvc >= 0.5).mean()), 3),
        "stage_seconds": {k: v for k, v in ev.stages.items()},
        "total_seconds": round(time.perf_counter() - t_all, 2),
        "model_resolution_hw": infer_stats["model_resolution_hw"],
        "peak_gpu_memory_gb": infer_stats["peak_gpu_memory_gb"],
        "measurement": "disabled (no independently validated scale)",
    }
    write_json(job_dir / "metrics.json", metrics)
    artifacts = ["cloud.ply", "cameras.json", "georef.json", "metrics.json", "frames.jsonl",
                 "input_manifest.json", "config.json", "environment.json"]
    write_json(job_dir / "artifact_manifest.json", {
        "artifacts": {a: {"sha256": sha256(job_dir / a), "bytes": (job_dir / a).stat().st_size} for a in artifacts},
        "cloud_ply_fields": "x y z (float32, coordinate_frame), red green blue, mv_consistency (fraction of "
                            "overlapping views whose predicted depth agrees; model self-consistency, not "
                            "independent triangulation), model_conf (learned confidence), view_id",
        "coordinate_frame": frame_name, "scale_source": scale_source, "geolocation": geolocation,
        "validation_status": "unvalidated", "backend": metrics["backend"],
    })
    ev.log(stage="pipeline", event="done", total_seconds=metrics["total_seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
