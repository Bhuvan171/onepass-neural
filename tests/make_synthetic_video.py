"""Build a SYNTHETIC test video + DJI-style SRT from the Brighton Beach photos.

Each photo is shown for 1 s; the SRT carries that photo's EXIF GPS. This exercises
video decoding, frame timestamps and SRT alignment only. It is not real drone
video and says nothing about blur, rolling shutter or real GPS synchronisation.
"""

import subprocess
from pathlib import Path

from onepass_neural.georef import read_exif_gps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic"


def fmt(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round(t % 1 * 1000)):03d}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted((ROOT / "data" / "brighton_beach" / "images").glob("*.JPG"))
    listing = OUT / "concat.txt"
    listing.write_text("".join(f"file '{p}'\nduration 1\n" for p in images) + f"file '{images[-1]}'\n")
    video = OUT / "brighton_SYNTHETIC_from_photos.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", "scale=1920:1080,fps=30", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(video)], check=True)
    blocks = []
    for i, p in enumerate(images):
        lat, lon, alt = read_exif_gps(p)
        blocks.append(f"{i + 1}\n{fmt(i)} --> {fmt(i + 1)}\n<font size=\"28\">FrameCnt: {i * 30 + 1}\n"
                      f"[latitude: {lat:.7f}] [longitude: {lon:.7f}] [rel_alt: 0.000 abs_alt: {alt:.3f}] </font>\n")
    (OUT / "brighton_SYNTHETIC_from_photos.srt").write_text("\n".join(blocks))
    print(video)


if __name__ == "__main__":
    main()
