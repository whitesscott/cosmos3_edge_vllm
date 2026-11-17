#!/usr/bin/env python3
"""Pull still frames out of an .mp4 for the PPE prompt sweep.

Default mode reproduces physical-ai-safety-observability's edge/frame_sampler.py
(`sample_opencv_frames`): cv2.VideoCapture reads frames sequentially, keeps every
frame where `index % stride == 0`, stops after `count` kept frames, and encodes each
with `cv2.imencode(".jpg", frame)` (OpenCV default quality 95, no resize, BGR in/out).
With --stride 60 --count 60 that matches examples/thor_video_source.json.

--spread N instead picks N frames evenly across the whole clip (stride/count only
cover the first stride*count frames). --png writes lossless frames, which keeps the
fine edges of glasses / vest reflective strips intact for small-object tests.

Writes <out>/frame_<index>.jpg|png plus manifest.json (index, timestamp, size).

    python extract_frames.py ludo_physical_game.mp4 -o frames/ --stride 60 --count 60
    python extract_frames.py clip.mp4 -o frames/ --spread 60 --png
"""

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="Input video file (or RTSP URI)")
    p.add_argument("-o", "--out", required=True, help="Output directory")
    p.add_argument("--stride", type=int, default=60, help="Keep frames where index %% stride == 0 (default 60)")
    p.add_argument("--count", type=int, default=60, help="Stop after this many kept frames (default 60)")
    p.add_argument("--spread", type=int, default=0, help="Instead: N frames evenly spaced over the whole clip")
    p.add_argument("--png", action="store_true", help="Lossless PNG instead of default-quality JPEG")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.spread:
        if total <= 0:
            raise SystemExit("--spread needs a known frame count (not available for this source)")
        n = min(args.spread, total)
        wanted = {round(i * (total - 1) / max(n - 1, 1)) for i in range(n)}
        keep = lambda i: i in wanted  # noqa: E731
        limit = len(wanted)
    else:
        keep = lambda i: i % args.stride == 0  # noqa: E731
        limit = args.count

    ext = ".png" if args.png else ".jpg"
    frames = []
    index = 0
    try:
        # Sequential read like the upstream sampler (no seeking: exact frame indices on H.264).
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if keep(index):
                ok, encoded = cv2.imencode(ext, frame)
                if not ok:
                    raise SystemExit(f"Encoding frame {index} failed")
                path = out / f"frame_{index:06d}{ext}"
                path.write_bytes(encoded.tobytes())
                h, w = frame.shape[:2]
                frames.append(
                    {"file": path.name, "frame_index": index, "t_sec": round(index / fps, 3) if fps else None,
                     "width": w, "height": h}
                )
            index += 1
    finally:
        capture.release()

    manifest = {
        "source": str(args.video),
        "fps": fps,
        "total_frames": total,
        "mode": f"spread {args.spread}" if args.spread else f"stride {args.stride} count {args.count}",
        "format": ext[1:],
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    span = f"{frames[0]['t_sec']}s–{frames[-1]['t_sec']}s" if frames and fps else "?"
    print(f"Wrote {len(frames)} frames ({span} of {total / fps if fps else 0:.1f}s) to {out}")


if __name__ == "__main__":
    main()
