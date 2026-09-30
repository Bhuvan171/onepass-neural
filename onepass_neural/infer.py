"""MapAnything (Apache checkpoint) inference: images -> per-view dense geometry.

Only the explicitly Apache-licensed checkpoint at a pinned revision is allowed.
If it cannot be loaded we fail; we never fall back to the CC-BY-NC checkpoint.
"""

import os
import time

import numpy as np
import torch

MODEL_ID = "facebook/map-anything-apache"
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
# Shared 80 GB A100: stay under ~38 GB for this process.
GPU_MEMORY_FRACTION = 0.48


def load_model(device="cuda"):
    from mapanything.models import MapAnything

    assert MODEL_ID == "facebook/map-anything-apache", "only the Apache checkpoint is allowed"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    model = MapAnything.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)
    model.eval()
    return model


def run_inference(model, image_paths, device="cuda"):
    """Run MapAnything on image_paths (in the given order). Returns (views, stats).

    views: list of dicts of numpy arrays, one per input image:
      pts3d (H,W,3) world frame = first camera's OpenCV frame, metres (learned scale)
      depth_z (H,W), intrinsics (3,3), cam2world (4,4), rgb (H,W,3) uint8,
      conf (H,W) learned confidence, mv_consistency (H,W) in [0,1], mask (H,W) bool
    """
    from mapanything.utils.image import load_images
    from mapanything.utils.multiview_confidence import compute_multiview_depth_confidence

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    inputs = load_images([str(p) for p in image_paths])
    t_load = time.perf_counter()

    with torch.no_grad():
        preds = model.infer(
            inputs,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            use_multiview_confidence=False,  # keep the learned confidence; consistency computed below
        )
        torch.cuda.synchronize()
        t_infer = time.perf_counter()

        # Cross-view consistency of the model's own predicted depths and poses.
        # This is NOT independent triangulation; it only says the predictions agree with each other.
        mv = compute_multiview_depth_confidence(
            depth_z=[p["depth_z"].float() for p in preds],
            intrinsics=[p["intrinsics"].float() for p in preds],
            camera_poses=[p["camera_poses"].float() for p in preds],
            depth_masks=[p["non_ambiguous_mask"].bool() for p in preds],
        )
        torch.cuda.synchronize()
    t_mv = time.perf_counter()

    views = []
    for p, mv_conf in zip(preds, mv):
        views.append({
            "pts3d": p["pts3d"][0].float().cpu().numpy(),
            "depth_z": p["depth_z"][0, ..., 0].float().cpu().numpy(),
            "intrinsics": p["intrinsics"][0].float().cpu().numpy(),
            "cam2world": p["camera_poses"][0].float().cpu().numpy(),
            "rgb": (p["img_no_norm"][0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            "conf": p["conf"][0].float().cpu().numpy(),
            "mv_consistency": mv_conf[0].float().cpu().numpy(),
            "mask": p["mask"][0, ..., 0].bool().cpu().numpy(),
            "metric_scaling_factor": float(p["metric_scaling_factor"].reshape(-1)[0]),
        })

    stats = {
        "num_views": len(views),
        "model_resolution_hw": list(views[0]["depth_z"].shape),
        "seconds_image_loading": round(t_load - t0, 3),
        "seconds_inference": round(t_infer - t_load, 3),
        "seconds_multiview_consistency": round(t_mv - t_infer, 3),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    return views, stats
