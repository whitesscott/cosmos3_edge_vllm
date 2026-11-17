#!/usr/bin/env python3
"""Build a PPE prompt-sweep image pool from a Roboflow-style COCO export.

Keeps images that contain people or head/torso PPE classes, copies them into
<out>/images/ (prefixed with the split), and writes <out>/labels.json:

    {"train__ppe_0750....jpg": {
        "split": "train", "width": 960, "height": 540,
        "coco_counts": {"hardhat": 4, "no-hardhat": 1},
        "coco_boxes": [{"label": "hardhat", "bbox_xywh": [x, y, w, h]}, ...],
        "verified": false,
        "truth": null}}

`coco_counts` come straight from the source annotations and are NOT reliable for
this dataset (person / vest boxes are not exhaustive, there is no glasses class,
some images are 4-tile mosaics). `truth` is filled in by hand (or in the lab UI)
with {"people": n, "hard_hat": n, "hi_vis_vest": n, "safety_glasses": n, ...};
the sweep only scores images where "verified" is true.

    python prep_coco_ppe.py ~/ppe-data/hf_construction_safety/extracted -o ~/ppe-data/construction_ppe
"""

import argparse
import collections
import json
import shutil
from pathlib import Path

PPE_CLASSES = {"person", "hardhat", "no-hardhat", "safety vest", "no-safety vest", "mask", "no-mask", "gloves"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="Directory holding <split>/_annotations.coco.json + images")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--splits", default="train,valid,test")
    p.add_argument("--min-side", type=int, default=400, help="Skip images whose short side is smaller")
    args = p.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.json"
    # Keep any hand-verified truth from a previous run.
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    kept = 0
    for split in args.splits.split(","):
        d = json.loads((Path(args.root) / split / "_annotations.coco.json").read_text())
        cats = {c["id"]: c["name"] for c in d["categories"]}
        anns = collections.defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            boxes = [{"label": cats[a["category_id"]], "bbox_xywh": a["bbox"]} for a in anns[im["id"]]]
            if not PPE_CLASSES & {b["label"] for b in boxes} or min(im["width"], im["height"]) < args.min_side:
                continue
            name = f"{split}__{im['file_name']}"
            shutil.copy2(Path(args.root) / split / im["file_name"], out / "images" / name)
            prev = labels.get(name, {})
            labels[name] = {
                "split": split,
                "width": im["width"],
                "height": im["height"],
                "coco_counts": dict(collections.Counter(b["label"] for b in boxes)),
                "coco_boxes": boxes,
                "verified": prev.get("verified", False),
                "truth": prev.get("truth"),
            }
            kept += 1

    labels_path.write_text(json.dumps(labels, indent=1))
    verified = sum(v["verified"] for v in labels.values())
    print(f"{kept} images -> {out / 'images'}; labels.json has {len(labels)} entries ({verified} verified)")


if __name__ == "__main__":
    main()
