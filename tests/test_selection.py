import cv2
import numpy as np

from onepass_neural import selection

SRT_NEW = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
[iso: 100] [latitude: 12.971600] [longitude: 77.594600] [rel_alt: 50.100 abs_alt: 912.300] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">FrameCnt: 2, DiffTime: 33ms
[iso: 100] [latitude: 12.971700] [longitude: 77.594700] [rel_alt: 50.200 abs_alt: 912.400] </font>
"""

SRT_OLD = """1
00:00:01,000 --> 00:00:02,000
HOME(77.5946,12.9716) 2020.01.01 10:00:00
GPS(77.5950,12.9720,15) BAROMETER:50.0
"""


def test_parse_dji_srt_formats():
    recs = selection.parse_dji_srt_text(SRT_NEW)
    assert len(recs) == 2 and recs[0]["lat"] == 12.9716 and recs[0]["alt"] == 912.3
    assert recs[1]["start"] == 0.033
    old = selection.parse_dji_srt_text(SRT_OLD)
    assert len(old) == 1 and old[0]["lat"] == 12.972 and old[0]["lon"] == 77.595


def test_gps_lookup_does_not_bridge_gaps():
    recs = selection.parse_dji_srt_text(SRT_OLD)  # covers 1.0-2.0 s only
    assert selection.gps_for_timestamp(recs, 1.5) is recs[0]
    assert selection.gps_for_timestamp(recs, 2.3) is recs[0]      # within 0.5 s
    assert selection.gps_for_timestamp(recs, 3.0) is None         # too far: no invented GPS


def test_quality_filter_rejects_blur(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for i in range(4):
        img = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        if i == 2:
            img = cv2.GaussianBlur(img, (31, 31), 10)
        p = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(p), img)
        frames.append({"path": p})
    out = selection.quality_filter(frames)
    assert [f["accepted"] for f in out] == [True, True, False, True]
    assert "blurred" in out[2]["reason"]
