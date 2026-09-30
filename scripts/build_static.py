#!/usr/bin/env python3
"""Export succeeded runs into a static site (site/) for GitHub Pages. No GPU or server needed to view it.

Usage: python scripts/build_static.py [run_id ...]   (default: every succeeded run except runs/_dev)
"""
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS, WEB, SITE = ROOT / "runs", ROOT / "web", ROOT / "site"
FILES = ["cloud.ply", "cameras.json", "artifact_manifest.json", "metrics.json", "georef.json"]


def jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def main():
    wanted = set(sys.argv[1:])
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "data").mkdir(parents=True)
    shutil.copytree(WEB / "vendor", SITE / "vendor")

    index = []
    for d in sorted(RUNS.glob("job_*")):
        state_p = d / "state.json"
        if not state_p.exists() or (wanted and d.name not in wanted):
            continue
        state = json.loads(state_p.read_text())
        if state.get("state") != "succeeded":
            continue
        out = SITE / "data" / d.name
        (out / "thumbs").mkdir(parents=True)
        for f in FILES:
            shutil.copy2(d / f, out / f)
        for t in (d / "thumbs").glob("*.jpg"):
            shutil.copy2(t, out / "thumbs" / t.name)
        job = dict(state, events=jsonl(d / "events.jsonl"), frames=jsonl(d / "frames.jsonl"),
                   metrics=json.loads((d / "metrics.json").read_text()),
                   georef=json.loads((d / "georef.json").read_text()))
        (out / "job.json").write_text(json.dumps(job))
        index.append({k: state[k] for k in ("id", "state", "created", "label")})
        print("exported", d.name, state["label"])

    if not index:
        sys.exit("no succeeded runs found")
    index.sort(key=lambda j: j["created"], reverse=True)
    (SITE / "data" / "jobs.json").write_text(json.dumps(index))
    html = (WEB / "index.html").read_text().replace('data-static="0"', 'data-static="1"', 1)
    (SITE / "index.html").write_text(html)
    (SITE / ".nojekyll").write_text("")
    print(f"site/ ready: {len(index)} run(s)")


if __name__ == "__main__":
    main()
