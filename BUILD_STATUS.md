# BUILD_STATUS — OnePass neural preview backend

**Date:** 30 Sep 2026 · **Machine:** NVIDIA A100 80GB PCIe (shared), driver 590.44.01, 64 CPU, 125 GB RAM,
Ubuntu 22.10, Python 3.10.7, torch 2.12.0+cu130 · **Process GPU cap:** 48% (~38 GB)

**Backend:** MapAnything, `facebook/map-anything-apache` revision `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`
(weights sha256 `fa06c0fd…d5201`), code commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
The CC-BY-NC `facebook/map-anything` checkpoint is not downloaded and cannot be selected.

**Sample:** Brighton Beach public photo set (github.com/pierotofy/drone_dataset_brighton_beach, BSD-2-Clause),
18 DJI FC300S photos, 4000×2250, EXIF GPS, 3 parallel flight lines. `images/` only; the repo's `model.laz` and
`dsm.tif` are not used and are not OnePass outputs. **This is a photo set, not drone video.**

## Measured runs (stored in `runs/`)

| Run | Input | Frames | Points exported | Total | Model load | Reconstruct | Peak GPU | GPS result |
|---|---|---|---|---|---|---|---|---|
| `job_…141548_51909e` | 18 photos, 3 lines | 18/18 | 1,134,767 | 29.8 s | 18.7 s | 5.9 s | 11.1 GB | applied; RMSE 1.43 m, LOO 1.62 m; GPS/model scale 3.96× |
| `job_…141328_5f11b5` | 6 photos, 1 line | 6/6 | 502,083 | 21.9 s | 18.3 s | 2.1 s | 7.0 GB | **degenerate** (s₂/s₁ 0.0094): not geolocated; scale 3.76× |
| `job_…142735_e37b84` | **synthetic** 1080p video made from the 18 photos + synthetic SRT | 18/18 | 1,025,708 | 30.8 s | 20.6 s | 3.7 s | 11.1 GB | applied; RMSE 0.66 m, LOO 0.89 m; scale 4.47× |
| `job_…142207_31d55c` | 1 uploaded photo | – | – | – | – | – | – | failed correctly: "Only 1 usable frame(s)" |
| `job_…142220_bd44e6` | 18 photos | – | – | – | – | – | – | cancelled while running; child process killed; no PLY served |

"Reconstruct" includes image loading, inference and the cross-view agreement computation. GPS residuals are
agreement with the input GPS, **not** accuracy. No independent checkpoints exist for this sample.

Other measurements: 4-view smoke test 0.86 s inference, 6.3 GB peak. Offline run inside a network-less
namespace (`unshare -rn`) completed: 1,134,767 points, 28.6 s.

## Findings worth keeping

- The model's learned metric scale is wrong on this aerial scene by ~4× (camera spacing 3.5 m predicted vs 13.7 m GPS),
  and it changes with input resolution (3.96× on photos, 4.47× on 1080p frames). GPS alignment is necessary; learned scale must not be trusted.
- Cross-view agreement is high on ground/roads and low on tree canopy, water and single-view edges (58% of exported points ≥ 0.5 on the 18-photo run).
- A single straight flight line cannot fix rotation about the line from GPS alone; the app refuses to geolocate it.

## Status

**Implemented and tested:** Apache checkpoint guard · MapAnything inference · masks + cross-view agreement per point ·
EXIF GPS · DJI SRT parser (unit tests + synthetic video only) · video decode · quality-filtered interval sampling ·
post-hoc GPS similarity with collinearity refusal · ENU export · PLY + manifests + hashes · job worker with cancel ·
viewer (orbit/pan/reset/point size/cameras/agreement colouring and filter/PLY download/stored runs) · offline operation ·
8 pytest checks passing.

**Not tested on real data:** real drone video, real DJI SRT (only synthetic), non-DJI telemetry.

**Failed / fixed during the build:** model construction fetches DINOv2 code from GitHub via torch.hub even with
`HF_HUB_OFFLINE=1` → cached in project `.torch/` by `scripts/setup.sh`, offline verified. PLYLoader method chaining
broke loading → fixed. Warning banner never displayed → fixed.

**Planned (not implemented):** parallax-aware selector (M4) · comparison with the COLMAP track on the same frames (M5) ·
COLMAP poses → MapAnything densification · independent accuracy evaluation (UseGeo / surveyed checkpoints) ·
LAS/GLB/OBJ/GeoTIFF exports · measurement tools · keeping the model loaded between jobs (would remove ~18 s cold start).

## Next gate

Run a **real drone video with its SRT** through the upload path. Pass = frames decode, SRT matches by timestamp,
GPS alignment is either applied with sensible residuals or correctly refused, and the cloud visibly matches the video.
Then M5: same frames through the COLMAP track and report inter-method agreement.
