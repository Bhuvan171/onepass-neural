"""Local web service: one worker thread runs one pipeline subprocess at a time.

Run: .venv/bin/uvicorn onepass_neural.server:app --host 127.0.0.1 --port 8765
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from onepass_neural.selection import IMAGE_EXTS, VIDEO_EXTS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WEB = ROOT / "web"
BB_IMAGES = sorted((ROOT / "data" / "brighton_beach" / "images").glob("*.JPG"))
BB_PROVENANCE = "github.com/pierotofy/drone_dataset_brighton_beach (BSD-2-Clause); images/ only"

SAMPLES = {
    "brighton_beach": {
        "images": [str(p) for p in BB_IMAGES],
        "label": "Public photo set: Brighton Beach, 18 images, 3 flight lines (not video)",
        "provenance": BB_PROVENANCE,
    },
    "brighton_beach_one_line": {
        "images": [str(p) for p in BB_IMAGES[:6]],
        "label": "Public photo set: Brighton Beach, first flight line only, 6 images (not video)",
        "provenance": BB_PROVENANCE,
    },
}

# Files a client may fetch from a job directory.
PUBLIC_FILES = {"cloud.ply", "cameras.json", "georef.json", "metrics.json", "input_manifest.json",
                "artifact_manifest.json", "config.json", "environment.json", "log.txt"}

app = FastAPI(title="OnePass neural preview")
jobs = {}
jobs_lock = threading.Lock()
work = queue.Queue()


def save_state(job):
    (RUNS / job["id"] / "state.json").write_text(json.dumps({k: v for k, v in job.items() if k != "proc"}, indent=2))


def set_state(job, state, **kw):
    with jobs_lock:
        job["state"] = state
        job.update(kw)
        save_state(job)


def load_existing_jobs():
    for d in sorted(RUNS.glob("job_*")):
        f = d / "state.json"
        if f.exists():
            job = json.loads(f.read_text())
            if job["state"] in ("queued", "running"):
                job["state"], job["error"] = "failed", "server restarted before the job finished"
                (d / "state.json").write_text(json.dumps(job, indent=2))
            jobs[job["id"]] = job


def worker():
    env = dict(os.environ, HF_HOME=str(ROOT / ".hf"), HF_HUB_OFFLINE="1", TORCH_HOME=str(ROOT / ".torch"))
    while True:
        job_id = work.get()
        job = jobs[job_id]
        if job["state"] == "cancelled":
            continue
        job_dir = RUNS / job_id
        with open(job_dir / "log.txt", "w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "onepass_neural.pipeline", str(job_dir)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            job["proc"] = proc
            set_state(job, "running", started=time.time())
            rc = proc.wait()
        job.pop("proc", None)
        if job["state"] == "cancelled":
            set_state(job, "cancelled", finished=time.time())
        elif rc == 0 and (job_dir / "artifact_manifest.json").exists():
            set_state(job, "succeeded", finished=time.time())
        else:
            tail = (job_dir / "log.txt").read_text(errors="ignore").strip().splitlines()[-3:]
            set_state(job, "failed", finished=time.time(), error=" | ".join(tail) or f"exit code {rc}")


@app.on_event("startup")
def startup():
    RUNS.mkdir(exist_ok=True)
    load_existing_jobs()
    threading.Thread(target=worker, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/vendor", StaticFiles(directory=WEB / "vendor"), name="vendor")


@app.get("/api/samples")
def samples():
    return {k: {"label": v["label"], "count": len(v["images"])} for k, v in SAMPLES.items()}


@app.post("/api/jobs")
async def create_job(sample: Optional[str] = Form(None), max_views: int = Form(40), sample_fps: float = Form(2.0),
                     files: Optional[List[UploadFile]] = File(None)):
    job_id = time.strftime("job_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    job_dir = RUNS / job_id
    spec = {"max_views": max(2, min(int(max_views), 200)), "sample_fps": max(0.1, min(float(sample_fps), 10.0))}
    if sample:
        if sample not in SAMPLES:
            raise HTTPException(400, f"unknown sample {sample}")
        spec.update(SAMPLES[sample])
        job_dir.mkdir(parents=True)
    elif files:
        inp = job_dir / "input"
        inp.mkdir(parents=True)
        saved = []
        for f in files:
            name = Path(f.filename or "").name
            ext = Path(name).suffix.lower()
            if not name or ext not in IMAGE_EXTS | VIDEO_EXTS | {".srt"}:
                raise HTTPException(400, f"unsupported file type: {name}")
            dest = inp / name
            dest.write_bytes(await f.read())
            dest.chmod(0o444)
            saved.append(dest)
        videos = [p for p in saved if p.suffix.lower() in VIDEO_EXTS]
        srts = [p for p in saved if p.suffix.lower() == ".srt"]
        images = sorted(p for p in saved if p.suffix.lower() in IMAGE_EXTS)
        if len(videos) == 1 and not images:
            spec.update(video=str(videos[0]), srt=str(srts[0]) if srts else None,
                        label=f"Uploaded video {videos[0].name}" + (" with SRT telemetry" if srts else " (no telemetry)"))
        elif images and not videos:
            spec.update(images=[str(p) for p in images], label=f"Uploaded image set ({len(images)} images)")
        else:
            raise HTTPException(400, "upload either one video (+ optional .srt) or a set of images")
        spec["provenance"] = "user upload"
    else:
        raise HTTPException(400, "choose a sample or upload files")
    (job_dir / "job.json").write_text(json.dumps(spec, indent=2))
    job = {"id": job_id, "state": "queued", "created": time.time(), "label": spec["label"]}
    with jobs_lock:
        jobs[job_id] = job
        save_state(job)
    work.put(job_id)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return sorted(({k: v for k, v in j.items() if k != "proc"} for j in jobs.values()),
                  key=lambda j: j["created"], reverse=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    d = RUNS / job_id
    out = {k: v for k, v in job.items() if k != "proc"}
    out["events"] = read_jsonl(d / "events.jsonl")
    out["frames"] = read_jsonl(d / "frames.jsonl")
    if job["state"] == "succeeded":
        out["metrics"] = json.loads((d / "metrics.json").read_text())
        out["georef"] = json.loads((d / "georef.json").read_text())
    return JSONResponse(out)


@app.get("/api/jobs/{job_id}/file/{name:path}")
def get_file(job_id: str, name: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    is_thumb = name.startswith("thumbs/") and name.count("/") == 1 and name.endswith(".jpg")
    if name not in PUBLIC_FILES and not is_thumb:
        raise HTTPException(404, "not a public artifact")
    if name == "cloud.ply" and job["state"] != "succeeded":
        raise HTTPException(409, f"job is {job['state']}; no result to show")
    path = (RUNS / job_id / name).resolve()
    if not path.is_file() or RUNS.resolve() not in path.parents:
        raise HTTPException(404, "not found")
    return FileResponse(path, filename=f"{job_id}_{Path(name).name}" if name == "cloud.ply" else None)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["state"] in ("queued", "running"):
        proc = job.get("proc")
        set_state(job, "cancelled")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return {"id": job_id, "state": job["state"]}
