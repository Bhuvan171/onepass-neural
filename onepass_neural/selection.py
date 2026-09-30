"""Input frames: image-set loading, video frame extraction, DJI SRT GPS, and
quality-filtered interval sampling.

The selector is deliberately simple and is NOT parallax-aware: it rejects
blurred / badly exposed frames, then keeps evenly spaced frames up to a budget.
"""

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def quality_scores(path, work_width=640):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    scale = work_width / img.shape[1]
    img = cv2.resize(img, (work_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return {
        "sharpness": float(cv2.Laplacian(img, cv2.CV_64F).var()),
        "brightness": float(img.mean()),
    }


def quality_filter(frames, rel_sharpness=0.25, dark=20, bright=235):
    """Mark frames accepted/rejected. Sharpness threshold is relative to the set median."""
    for f in frames:
        f["quality"] = quality_scores(f["path"])
    sharp = [f["quality"]["sharpness"] for f in frames if f["quality"]]
    min_sharp = rel_sharpness * float(np.median(sharp)) if sharp else 0.0
    for f in frames:
        q = f["quality"]
        if q is None:
            f["accepted"], f["reason"] = False, "unreadable image"
        elif q["sharpness"] < min_sharp:
            f["accepted"], f["reason"] = False, f"blurred (sharpness {q['sharpness']:.0f} < {min_sharp:.0f}, 25% of set median)"
        elif q["brightness"] < dark:
            f["accepted"], f["reason"] = False, "too dark"
        elif q["brightness"] > bright:
            f["accepted"], f["reason"] = False, "overexposed"
        else:
            f["accepted"], f["reason"] = True, "passed quality filter"
    return frames


def interval_sample(frames, max_views):
    """Among accepted frames keep up to max_views evenly spaced ones."""
    acc = [f for f in frames if f["accepted"]]
    if len(acc) <= max_views:
        return frames
    keep = set(np.round(np.linspace(0, len(acc) - 1, max_views)).astype(int).tolist())
    for i, f in enumerate(acc):
        if i not in keep:
            f["accepted"], f["reason"] = False, f"not sampled (frame budget {max_views})"
    return frames


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def probe_video(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,codec_name:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    import json
    info = json.loads(out)
    s = info["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": s["width"], "height": s["height"], "codec": s["codec_name"],
        "fps": float(num) / float(den), "duration_s": float(info["format"]["duration"]),
    }


def extract_video_frames(video, out_dir, fps):
    """Decode at a fixed sampling rate. Frame k is at k/fps seconds from stream start."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
         str(out_dir / "frame_%05d.jpg")],
        check=True,
    )
    paths = sorted(out_dir.glob("frame_*.jpg"))
    return [{"path": p, "timestamp_s": round(i / fps, 3)} for i, p in enumerate(paths)]


def _srt_time(s):
    h, m, rest = s.split(":")
    sec, ms = rest.replace(".", ",").split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def parse_dji_srt(path):
    return parse_dji_srt_text(Path(path).read_text(errors="ignore"))


def parse_dji_srt_text(text):
    """Parse DJI-style SRT telemetry. Returns list of {start, end, lat, lon, alt, alt_type}.

    Supports the bracketed format ([latitude: ..] [longitude: ..] [rel_alt: .. abs_alt: ..])
    and the older GPS(lon,lat,alt) format. Other layouts are not guessed.
    """
    records = []
    for block in re.split(r"\n\s*\n", text.strip()):
        tm = re.search(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", block)
        if not tm:
            continue
        lat = re.search(r"latitude\s*:\s*(-?[\d.]+)", block)
        lon = re.search(r"longitude\s*:\s*(-?[\d.]+)", block)
        abs_alt = re.search(r"abs_alt\s*:\s*(-?[\d.]+)", block)
        rel_alt = re.search(r"rel_alt\s*:\s*(-?[\d.]+)", block)
        old = re.search(r"GPS\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", block)
        if lat and lon:
            la, lo = float(lat.group(1)), float(lon.group(1))
            alt, alt_type = (float(abs_alt.group(1)), "abs_alt (datum unknown)") if abs_alt else \
                ((float(rel_alt.group(1)), "rel_alt (takeoff-relative)") if rel_alt else (0.0, "missing"))
        elif old:
            lo, la, alt, alt_type = float(old.group(1)), float(old.group(2)), float(old.group(3)), "GPS() altitude (datum unknown)"
        else:
            continue
        if la == 0 and lo == 0:
            continue  # no fix
        records.append({"start": _srt_time(tm.group(1)), "end": _srt_time(tm.group(2)),
                        "lat": la, "lon": lo, "alt": alt, "alt_type": alt_type})
    return records


def gps_for_timestamp(records, t, max_gap_s=0.5):
    """SRT record covering time t, or the nearest one within max_gap_s. No interpolation."""
    best, best_d = None, None
    for r in records:
        d = 0.0 if r["start"] <= t < r["end"] else min(abs(t - r["start"]), abs(t - r["end"]))
        if best_d is None or d < best_d:
            best, best_d = r, d
    return best if best is not None and best_d <= max_gap_s else None
