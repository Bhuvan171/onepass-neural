# Claims ledger — neural preview backend

| Statement you may make | Status | Evidence | Limitation |
|---|---|---|---|
| OnePass has a second, neural reconstruction backend using an Apache-2.0 MapAnything checkpoint | implemented | `onepass_neural/infer.py`, `MODEL_LICENSES.md` | model prediction, not measured geometry |
| 18 aerial photos → 1.13M-point coloured cloud in 29.8 s end to end (5.9 s reconstruction, 18.7 s model load) on an A100 | measured | `runs/job_…141548_51909e/metrics.json` | photo set, not a 10-minute video; one sample |
| Peak GPU memory 11.1 GB for 18 views | measured | same | A100; not measured on other GPUs |
| Output is aligned to GPS; GPS agreement RMSE 1.43 m (leave-one-out 1.62 m) | measured | `georef.json` | agreement with input GPS, **not** spatial accuracy |
| The system refuses to geolocate a single straight flight line and says why | implemented | `runs/job_…141328_5f11b5/georef.json` | – |
| The model's own metric scale was ~4× off on this scene; GPS corrects the scale | measured | `georef.json` scale 3.96× | one scene |
| Per-point cross-view agreement is shown and can hide disagreeing points | implemented | viewer | self-consistency of predictions, not accuracy or triangulation |
| Runs fully offline once provisioned | measured | network-less namespace run, 28.6 s | needs `scripts/setup.sh` once with network |
| Accepts drone video + DJI SRT | implemented, **synthetic test only** | `job_…142735_e37b84` | no real drone video tested yet |

**Do not say:** "≤1 m accuracy", "real-time", "complete model", "measurement-grade", "parallax-aware frame selection",
"all permissive licences", "processes a 10-minute video in X minutes", or "adaptive" selection. None of these is supported yet.
