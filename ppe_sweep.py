#!/usr/bin/env python3
"""Headless PPE question-wording sweep against a Cosmos3-Edge vLLM server.

Runs every (strategy x variant x thinking) combination over the labelled images and
scores them the way the forum benchmark does: per image and item, a *real alarm* is
a person truly not wearing it; a *kept alarm* is the model also reporting a "no";
a *false alarm* is the model reporting a "no" when everyone truly wears it.

    strategy  full   one question about the whole frame
              crops  ask for person boxes, then crop + enlarge each person and ask
                     about that one person (small gear gets many more pixels)
    thinking  off    structured JSON (xgrammar), --max-tokens-nothink (512)
              on     free text, JSON parsed after </think>; sampling 0.6/0.95/20
                     (greedy + thinking loops to the token cap on this model)

Labels: <data>/labels.json, entries with "truth" (see prep_coco_ppe.py). Only
"verified": true entries are scored unless --include-drafts.

    python ppe_sweep.py --data ~/ppe-data/construction_ppe --include-drafts
    python ppe_sweep.py --variants V0_bool,V4_region_evidence --strategies full --thinking off
    python ppe_sweep.py --dry-run          # print prompts + schemas only

Writes <out>/results.jsonl (every model call), <out>/summary.json and <out>/report.md.
"""

import argparse
import base64
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image

import ppe_prompts as P

THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(Path.home() / "ppe-data/construction_ppe"))
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="nvidia/Cosmos3-Edge")
    p.add_argument("--variants", default=",".join(P.VARIANTS))
    p.add_argument("--strategies", default="full,crops")
    p.add_argument("--thinking", default="off,on", help="off, on, or off,on")
    p.add_argument("--include-drafts", action="store_true", help="Also score unverified draft labels")
    p.add_argument("--limit", type=int, default=0, help="Only the first N images")
    p.add_argument("--max-tokens-nothink", type=int, default=512)
    p.add_argument("--max-tokens-think", type=int, default=4096)
    p.add_argument("--box-scale", default="1000", help="Detection box units: 1000 (normalized) or pixel")
    p.add_argument("--crop-margin", type=float, default=0.15, help="Pad each person box by this fraction")
    p.add_argument("--crop-min-side", type=int, default=768, help="Upscale crops so the long side is at least this")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--out", default=None, help="Output dir (default <data>/sweeps/<timestamp>)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ── vLLM client ──────────────────────────────────────────────────────────────


def image_data_url(img: Image.Image, max_pixels: int = 8_000_000) -> str:
    """JPEG data URL, downscaled to at most max_pixels. The server accepts up to ~16.7 MP
    (~16k vision tokens at 32x32 px/token), which alone fills the 16k context, after which
    vLLM silently clamps max_tokens to the few tokens left. 8 MP ≈ 8k tokens."""
    if img.width * img.height > max_pixels:
        f = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(args, img: Image.Image, prompt: str, schema: dict | None, think: bool) -> dict:
    """One request. Returns {raw, answer, thinking, finish, latency, tokens, error}."""
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(img)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": args.max_tokens_think if think else args.max_tokens_nothink,
        "chat_template_kwargs": {"enable_thinking": think},
        "seed": 0,
    }
    if think:
        body.update(THINK_SAMPLING)
    else:
        body["temperature"] = 0.0
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            }
    req = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            resp = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError) as e:
        detail = e.read().decode()[:500] if isinstance(e, urllib.error.HTTPError) else ""
        return {"error": f"{e} {detail}".strip(), "latency": time.time() - t0}
    latency = time.time() - t0
    choice = resp["choices"][0]
    msg = choice["message"]
    raw = msg.get("content") or ""
    thinking, answer = P.strip_thinking(raw)
    thinking = thinking or msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {
        "raw": raw,
        "answer": answer,
        "thinking": thinking,
        "finish": choice.get("finish_reason"),
        "latency": latency,
        "tokens": (resp.get("usage") or {}).get("completion_tokens"),
    }


# ── Strategies ───────────────────────────────────────────────────────────────


def to_pixels(box, w, h, scale: str):
    x1, y1, x2, y2 = box
    if scale == "1000":
        x1, x2, y1, y2 = x1 * w / 1000, x2 * w / 1000, y1 * h / 1000, y2 * h / 1000
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    return [x1, y1, x2, y2]


def crop_person(img: Image.Image, box, margin: float, min_side: int) -> Image.Image:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    c = img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))
    long_side = max(c.size)
    if long_side < min_side:
        f = min_side / long_side
        c = c.resize((max(1, round(c.width * f)), max(1, round(c.height * f))), Image.Resampling.LANCZOS)
    return c


def ask(args, img, variant, strat, think) -> list[dict]:
    """The variant's question for one image (full frame or one crop); 1 call, or 2 for two-step."""
    calls = []
    prompt = P.prompt_for(variant, strat, with_schema_hint=think)
    if "describe_full" in P.VARIANTS[variant]:
        dprompt = P.VARIANTS[variant][f"describe_{strat}"]
        d = chat(args, img, dprompt, None, think)
        d.update(stage=f"{strat}-describe", prompt=dprompt)
        calls.append(d)
        if "error" in d:
            return calls
        prompt = P.two_step_prompt(variant, strat, d.get("answer", ""), with_schema_hint=think)
    call = chat(args, img, prompt, P.schema_for(variant, strat), think)
    call.update(stage=strat, prompt=prompt)
    calls.append(call)
    return calls


def run_full(args, img, variant, think) -> tuple[list[dict] | None, list[dict]]:
    calls = ask(args, img, variant, "full", think)
    last = calls[-1]
    people = P.parse_people(P.extract_json(last.get("answer", ""))) if "error" not in last else None
    if people and all("bbox_2d" in p for p in people):
        # Grounded answer: drop phantom entries the same way as detection boxes.
        kept = P.filter_boxes([p["bbox_2d"] for p in people if P.parse_boxes([p])])
        survivors = []
        for p in people:
            b = P.parse_boxes([p])
            clamped = [max(0.0, min(1000.0, v)) for v in b[0]] if b else None
            if clamped in kept and clamped not in [s["_box"] for s in survivors]:
                p["_box"] = clamped
                survivors.append(p)
        for p in survivors:
            p["bbox_px"] = [round(v) for v in to_pixels(p.pop("_box"), img.width, img.height, "1000")]
        last["filtered_out"] = len(people) - len(survivors)
        people = survivors
    return people, calls


def run_crops(args, img, variant, think, box_cache: dict) -> tuple[list[dict] | None, list[dict]]:
    calls = []
    # Detection depends only on thinking mode, not on the wording variant: run it once.
    if think not in box_cache:
        prompt = P.detect_prompt()
        call = chat(args, img, prompt, P.DETECT_SCHEMA, think)
        call.update(stage="detect", prompt=prompt)
        boxes = P.parse_boxes(P.extract_json(call.get("answer", ""))) if "error" not in call else None
        call["boxes_raw"] = boxes
        if boxes is not None and args.box_scale == "1000":
            boxes = P.filter_boxes(boxes)
        call["boxes"] = boxes
        box_cache[think] = boxes
        calls.append(call)
    boxes = box_cache[think]
    if boxes is None:
        return None, calls
    people = []
    for i, b in enumerate(boxes):
        px = to_pixels(b, img.width, img.height, args.box_scale)
        if px[2] - px[0] < 4 or px[3] - px[1] < 4:
            continue
        crop_calls = ask(args, crop_person(img, px, args.crop_margin, args.crop_min_side), variant, "crop", think)
        for c in crop_calls:
            c.update(stage=f"crop{i}-{c['stage']}", box_px=[round(v) for v in px])
        calls += crop_calls
        last = crop_calls[-1]
        obj = P.extract_json(last.get("answer", "")) if "error" not in last else None
        person = P.parse_person(obj) if isinstance(obj, dict) else {k: "unclear" for k in P.ITEMS}
        person["bbox_px"] = last["box_px"]
        people.append(person)
    return people, calls


# ── Scoring ──────────────────────────────────────────────────────────────────


def score_image(truth: dict, people: list[dict] | None) -> dict:
    s: dict = {"parsed": people is not None}
    pred = people or []
    s["pred_people"] = len(pred)
    s["true_people"] = truth["people"]
    s["people_exact"] = truth.get("people_exact", True)
    for item in P.ITEMS:
        y, n, na = truth[item]
        counts = {a: sum(p[item] == a for p in pred) for a in ("yes", "no", "unclear")}
        real = n > 0
        scorable = real or na == 0  # no violation hidden behind "can't tell"
        alarm = counts["no"] > 0
        s[item] = {
            "scorable": scorable,
            "real": real,
            "kept": scorable and real and alarm,
            "false": scorable and not real and alarm,
            "pred": counts,
            "truth": [y, n, na],
        }
    return s


def summarize(rows: list[dict]) -> dict:
    out: dict = {"images": len(rows)}
    out["parse_fail"] = sum(not r["score"]["parsed"] for r in rows)
    calls = [c for r in rows for c in r["calls"]]
    lat = [c["latency"] for c in calls if "latency" in c]
    out["calls"] = len(calls)
    out["errors"] = sum("error" in c for c in calls)
    out["truncated"] = sum(c.get("finish") == "length" for c in calls)
    out["latency_image_median"] = statistics.median([sum(c.get("latency", 0) for c in r["calls"]) for r in rows]) if rows else 0
    out["latency_call_median"] = statistics.median(lat) if lat else 0
    out["latency_call_p95"] = sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else 0
    ex = [r for r in rows if r["score"]["people_exact"]]
    out["people_found"] = sum(min(r["score"]["pred_people"], r["score"]["true_people"]) for r in rows)
    out["people_true"] = sum(r["score"]["true_people"] for r in rows)
    out["people_abs_err_exact"] = (
        statistics.mean(abs(r["score"]["pred_people"] - r["score"]["true_people"]) for r in ex) if ex else None
    )
    for item in P.ITEMS:
        its = [r["score"][item] for r in rows]
        answers = sum(sum(i["pred"].values()) for i in its)
        out[item] = {
            "real_alarms": sum(i["scorable"] and i["real"] for i in its),
            "kept": sum(i["kept"] for i in its),
            "false": sum(i["false"] for i in its),
            "clean_images": sum(i["scorable"] and not i["real"] for i in its),
            "unclear_rate": (sum(i["pred"]["unclear"] for i in its) / answers) if answers else 0,
        }
    return out


def write_report(out_dir: Path, summary: dict, meta: dict):
    lines = [
        f"# PPE wording sweep — {meta['started']}",
        "",
        f"Model `{meta['model']}` at `{meta['endpoint']}` · {meta['n_images']} images "
        f"({'incl. unverified drafts' if meta['include_drafts'] else 'verified only'}) · data `{meta['data']}`",
        "",
        "Kept = real alarms the model also raised; False = alarms on images where everyone wears the item.",
        "",
        "| run | people found | hard hat kept / real | vest kept / real | glasses kept / real "
        "| false (hat/vest/glasses) | unclear % | parse fail | trunc | median s/image |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run, s in summary.items():
        hh, v, g = (s[i] for i in P.ITEMS)
        unclear = 100 * statistics.mean(s[i]["unclear_rate"] for i in P.ITEMS)
        lines.append(
            f"| {run} | {s['people_found']}/{s['people_true']} | {hh['kept']}/{hh['real_alarms']} | "
            f"{v['kept']}/{v['real_alarms']} | {g['kept']}/{g['real_alarms']} | "
            f"{hh['false']}/{v['false']}/{g['false']} (of {hh['clean_images']}/{v['clean_images']}/{g['clean_images']}) | "
            f"{unclear:.0f} | {s['parse_fail']} | {s['truncated']} | {s['latency_image_median']:.1f} |"
        )
    lines += ["", "Raw calls: `results.jsonl` (prompt, thinking, answer, parsed people per image)."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    variants = [v for v in args.variants.split(",") if v]
    strategies = [s for s in args.strategies.split(",") if s]
    modes = [{"off": False, "on": True}[m] for m in args.thinking.split(",") if m]
    for v in variants:
        if v not in P.VARIANTS:
            sys.exit(f"Unknown variant {v}; choose from {', '.join(P.VARIANTS)}")

    if args.dry_run:
        for v in variants:
            for strat in ("full", "crop"):
                print(f"===== {v} / {strat}\n{P.prompt_for(v, strat, with_schema_hint=False)}")
                print("schema:", json.dumps(P.schema_for(v, strat)))
        print(f"===== detect (no schema)\n{P.DETECT_PROMPT}")
        return

    data = Path(args.data)
    labels = json.loads((data / "labels.json").read_text())
    items = [
        (name, e) for name, e in sorted(labels.items())
        if e.get("truth") and (e.get("verified") or args.include_drafts)
    ]
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit("No labelled images to score (use --include-drafts for draft labels).")

    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else data / "sweeps" / started
    out_dir.mkdir(parents=True, exist_ok=True)
    results = (out_dir / "results.jsonl").open("w")
    runs = [(s, v, t) for s in strategies for t in modes for v in variants]
    print(f"{len(items)} images x {len(runs)} runs -> {out_dir}")

    rows_by_run: dict[str, list] = {f"{s}/{v}/think-{'on' if t else 'off'}": [] for s, v, t in runs}
    for k, (name, entry) in enumerate(items, 1):
        img = Image.open(data / "images" / name).convert("RGB")
        box_cache: dict = {}
        for strat, variant, think in runs:
            run = f"{strat}/{variant}/think-{'on' if think else 'off'}"
            if strat == "full":
                people, calls = run_full(args, img, variant, think)
            else:
                people, calls = run_crops(args, img, variant, think, box_cache)
            score = score_image(entry["truth"], people)
            row = {"run": run, "image": name, "people": people, "score": score, "calls": calls}
            results.write(json.dumps(row) + "\n")
            results.flush()
            rows_by_run[run].append(row)
            errs = [c["error"] for c in calls if "error" in c]
            flag = f" ERROR {errs[0][:120]}" if errs else ""
            print(f"[{k}/{len(items)}] {name[:40]:40s} {run:40s} people {score['pred_people']}/{score['true_people']}"
                  f" {sum(c.get('latency', 0) for c in calls):5.1f}s{flag}", flush=True)

    summary = {run: summarize(rows) for run, rows in rows_by_run.items()}
    meta = {
        "started": started, "model": args.model, "endpoint": args.endpoint, "data": str(data),
        "n_images": len(items), "include_drafts": args.include_drafts, "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "runs": summary}, indent=1))
    write_report(out_dir, summary, meta)
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
