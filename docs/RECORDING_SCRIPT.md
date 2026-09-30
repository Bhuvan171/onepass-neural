# Recording script — neural preview segment (~75 s)

Before recording: `scripts/serve.sh`, open http://127.0.0.1:8765, browser full screen.
Do not cut out processing without an on-screen "time cut — measured duration: N s" label.

**0–10 s.** Show the input selector. Say: "Input: a public drone photo set — 18 photos over Brighton Beach,
with GPS in each photo. This is a photo set, not our own flight video."

**10–25 s.** Press Reconstruct. The job goes queued → running; stages appear as they finish. Say: "Frames are
quality-filtered. The reconstruction is MapAnything, using its Apache-licensed weights, and it runs locally with no network."

**25–40 s.** Wait for success (~30 s; or cut with the label). Point at the stage times: "Model load about 19 seconds,
reconstruction about 6 seconds, peak GPU memory 11 GB, all taken from this run's log."

**40–60 s.** Orbit the cloud; show the cameras above the scene; switch to "cross-view agreement": "Blue is where
the model's predictions agree across photos; amber is where they don't: tree canopy, water, edges seen once.
That's the model agreeing with itself, not accuracy."

**60–70 s.** Scroll to GPS alignment: "The model's own scale was about 4 times too small. We align it to the
photos' GPS: agreement 1.4 metres RMSE. That is agreement with GPS, not measured ground accuracy."
Optionally load the stored single-line run: "With one straight flight line, the rotation can't be determined
from GPS, so OnePass refuses to place it on the map."

**70–80 s.** Download PLY. Say: "Working today: import, reconstruction, GPS alignment with refusal checks, viewer,
PLY export. Next: real drone video with telemetry, comparison against our COLMAP pipeline, and independent
accuracy testing against the 1-metre target."

Screenshot for the slide: `docs/screenshots/full_18_images_rgb.png` (genuine, from run `job_…141548_51909e`).
