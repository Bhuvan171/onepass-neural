# Licence inventory (components actually used)

| Component | Version / revision | Licence | Source | Decision |
|---|---|---|---|---|
| MapAnything code | commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | Apache-2.0 | github.com/facebookresearch/map-anything (`third_party/map-anything/LICENSE`) | used |
| MapAnything weights | `facebook/map-anything-apache` @ `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`, sha256 `fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201` | Apache-2.0 (HF model card) | huggingface.co/facebook/map-anything-apache | used |
| `facebook/map-anything` weights | – | CC-BY-NC 4.0 | – | **excluded**, never downloaded |
| MapAnything `[all]` extras (VGGT, Pi3, MASt3R, …) | – | mixed / non-commercial | – | **not installed** |
| DINOv2 code (fetched by torch.hub at model build) | `facebookresearch/dinov2` main branch, cached in `.torch/hub` (unpinned) | Apache-2.0 | github.com/facebookresearch/dinov2 | used; pin this later |
| UniCeption | 0.1.7 | BSD-3-Clause (package metadata) | PyPI | used (MapAnything dependency) |
| PyTorch / torchvision | 2.12.0+cu130 / 0.27.0 | BSD-3-Clause | PyPI | used |
| OpenCV (headless) | 4.10.0.84 | Apache-2.0 | PyPI | used |
| Three.js (+ OrbitControls, PLYLoader) | 0.160.0, vendored in `web/vendor/` | MIT (`web/vendor/LICENSE`) | npm via jsDelivr | used |
| FastAPI / Uvicorn / plyfile / ExifRead | see `requirements.lock.txt` | MIT / BSD-3 / GPL-3.0+ (plyfile) / BSD | PyPI | plyfile is GPL-3.0+ and is also a MapAnything core dependency: review before distribution |
| FFmpeg | system 5.1.1 (Ubuntu build) | Ubuntu build is GPL-enabled | apt | used for video decode only; review before distribution |
| Brighton Beach images | repo HEAD at clone time | BSD-2-Clause | github.com/pierotofy/drone_dataset_brighton_beach | demo sample, attribution kept |

The stack is **not** all-permissive: plyfile (GPL-3.0) and the Ubuntu FFmpeg build must be replaced or reviewed
before any claim that every component is MIT/BSD/Apache.
