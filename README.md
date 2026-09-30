# OnePass — neural preview backend (sixth harmony, SIH26158)

Second reconstruction method for OnePass, built alongside the COLMAP (geometric) track.
Drone photos or video (+ optional DJI SRT telemetry) → selected frames → **MapAnything**
feed-forward reconstruction (Apache-2.0 checkpoint) → post-hoc GPS alignment → coloured
point cloud in a local browser viewer, with PLY download and run manifests.

**Live demo (static): https://bhuvan171.github.io/onepass-neural/**: an interactive viewer of pre-computed results, see [Limitations](#limitations).

Every result in the live app is labelled **"AI preview — spatial accuracy not validated"**. Measurement is disabled.

## Run it

```bash
scripts/setup.sh      # once, needs network: venv, pinned MapAnything, weights, sample data
scripts/serve.sh      # offline afterwards; open http://127.0.0.1:8765
```

From VS Code Remote, forward port 8765. Pick a sample (or upload images / one video + .srt),
press **Reconstruct**. A run takes ~22–31 s on the A100, of which ~18–21 s is model loading (cold start per job).

Command-line run without the server:

```bash
mkdir -p runs/my_run && echo '{"images": ["/abs/path/a.jpg", "/abs/path/b.jpg"], "label": "my test"}' > runs/my_run/job.json
HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 TORCH_HOME=$PWD/.torch .venv/bin/python -m onepass_neural.pipeline runs/my_run
```

Video job: `{"video": "/abs/clip.mp4", "srt": "/abs/clip.srt", "sample_fps": 2, "max_views": 40}`.

Tests (GPU needed for the integration test): `HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 TORCH_HOME=$PWD/.torch .venv/bin/python -m pytest -q tests`

## How it works

| Stage | Module | What it really does |
|---|---|---|
| Ingest + select | `selection.py` | Image set, or ffmpeg decode at fixed fps (timestamp = index/fps). Rejects blurred (< 25% of set median Laplacian variance) and badly exposed frames, then evenly samples up to the frame budget. **Not parallax-aware.** |
| Telemetry | `selection.py`, `georef.py` | EXIF GPS, or DJI SRT (bracketed or old `GPS(...)` format). SRT record matched by timestamp, max 0.5 s gap, no interpolation. |
| Reconstruct | `infer.py` | `facebook/map-anything-apache` @ `00f9c245`, bf16, memory-efficient inference, upstream masks + edge masking. Process capped at 48% of GPU memory. Also computes cross-view depth agreement of the model's own predictions. |
| GPS alignment | `georef.py` | Umeyama similarity + RANSAC from predicted camera centres to GPS in local ENU. If the GPS track is near-collinear (s₂/s₁ < 0.1) no placement is applied; only the GPS-derived scale. Reports residuals and leave-one-out residuals (agreement with input GPS, not accuracy). |
| Fuse + export | `fuse.py`, `pipeline.py` | Voxel downsample (larger horizontal extent / 600), binary PLY with `mv_consistency`, `model_conf`, `view_id` per point. |
| Serve | `server.py`, `web/index.html` | One worker, one subprocess per job (cancel kills it). States: queued/running/succeeded/failed/cancelled. A job's PLY is only served if that job succeeded. Three.js is vendored (no CDN). |

## Run folder (`runs/<job_id>/`)

`job.json` (request) · `state.json` · `log.txt` · `events.jsonl` (stage start/end, seconds) · `input_manifest.json`
(input hashes, provenance) · `frames.jsonl` (every candidate, accept/reject reason, GPS) · `thumbs/` ·
`config.json` · `environment.json` (GPU, driver, torch, MapAnything commit, model revision) · `georef.json` ·
`cameras.json` · `cloud.ply` · `metrics.json` · `artifact_manifest.json` (sha256 of every artifact, frame/scale/validation status).

## Coordinates

- Georeferenced run: local ENU metres (x east, y north, z up) about `georef.json: enu_origin_lat_lon_alt`. Vertical datum unknown (EXIF/SRT altitude).
- Otherwise: MapAnything world frame = first camera, OpenCV axes (x right, y down, z forward), learned scale (or GPS spread-ratio scale on a straight-line track).
- MapAnything returns cam2world poses, so the camera centre is the pose translation.
- Viewer mapping: ENU (E,N,U) → three.js (E,U,−N); model frame (x,y,z) → (x,−z,y); centred before rendering.

See `BUILD_STATUS.md` for measured results and `CLAIMS.md` before writing any slide text.

## Limitations

- **AI preview — spatial accuracy not validated.** The geometry is a neural-network prediction. No independent accuracy validation (survey checkpoints, ground truth) has been done, GPS residuals only show agreement with the input GPS, and measurement tools are disabled. The altitude datum is unknown.
- **The hosted demo is not running the reconstruction.** Reconstruction needs an NVIDIA GPU (developed and measured on an A100), which free hosting does not provide. The GitHub Pages site is a static viewer of outputs from successful reconstructions run beforehand on the GPU machine (the Brighton Beach sample photos and a synthetic video made from them). You cannot upload your own images or video there. To run it for real, use `scripts/setup.sh` and `scripts/serve.sh` on a GPU machine.
- The demo runs are a public photo set and a *synthetic* video built from it, not real drone video. See [BUILD_STATUS.md](BUILD_STATUS.md) for measured results.
- Learned metric scale is unreliable on aerial scenes (~4× off on the sample), so GPS alignment is required for usable scale. Frame selection is not parallax-aware.
