#!/usr/bin/env python3
"""PPE prompt lab: try question wordings on Cosmos3-Edge interactively, review labels.

Needs the vLLM server from serve_cosmos3_edge_vllm.sh. Uses the same prompts,
schemas, crop logic and parsing as ppe_sweep.py, so what you see here is what
the sweep scores.

    ~/.git/uv/cosmos-vllm/bin/python ppe_lab.py [--data ~/ppe-data/construction_ppe] [--port 7861]
"""

import argparse
import json
import types
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

import ppe_prompts as P
import ppe_sweep as S

COLORS = {"yes": (40, 200, 40), "no": (230, 40, 40), "unclear": (240, 180, 0)}
SHORT = {"hard_hat": "H", "hi_vis_vest": "V", "safety_glasses": "G"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(Path.home() / "ppe-data/construction_ppe"))
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="nvidia/Cosmos3-Edge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--auth", default=None, help="user:password (use with a public tunnel)")
    return p.parse_args()


ARGS = parse_args()
DATA = Path(ARGS.data)
LABELS_PATH = DATA / "labels.json"


def load_labels() -> dict:
    return json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}


def labelled_names() -> list[str]:
    """Drafts awaiting review first, then verified, then unlabelled."""
    L = load_labels()
    key = lambda n: (0 if L[n].get("truth") and not L[n].get("verified") else 1 if L[n].get("verified") else 2, n)  # noqa: E731
    return sorted(L, key=key)


def client_args(max_nothink: int, max_think: int, crop_min_side: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model=ARGS.model, endpoint=ARGS.endpoint, timeout=300, box_scale="1000",
        max_tokens_nothink=int(max_nothink), max_tokens_think=int(max_think),
        crop_margin=0.15, crop_min_side=int(crop_min_side),
    )


# ── Try a wording ────────────────────────────────────────────────────────────


def default_prompt(variant: str, strategy: str, think: bool) -> str:
    return P.prompt_for(variant, "crop" if strategy == "crops" else "full", with_schema_hint=think)


def pick_image(name: str | None):
    return str(DATA / "images" / name) if name else None


def draw(img: Image.Image, people: list[dict]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.width // 300)
    for i, p in enumerate(people, 1):
        box = p.get("bbox_px")
        if not box:
            continue
        bad = any(p[k] == "no" for k in P.ITEMS)
        d.rectangle(box, outline=COLORS["no" if bad else "yes"], width=lw)
        x, y = box[0] + lw, box[1] + lw
        d.text((x, y), f"#{i}", fill=(255, 255, 255))
        for k in P.ITEMS:
            x += 14
            d.text((x + 10, y), SHORT[k], fill=COLORS[p[k]])
    return out


def run(image_path, strategy, variant, think, prompt, max_nothink, max_think, crop_min_side):
    if not image_path:
        raise gr.Error("Pick or upload an image first.")
    img = Image.open(image_path).convert("RGB")
    args = client_args(max_nothink, max_think, crop_min_side)
    strat = "crop" if strategy == "crops" else "full"
    # The editable prompt replaces the variant's wording (the schema hint is re-added for
    # thinking mode); the answer schema and two-step describe call stay the variant's.
    orig = P.VARIANTS[variant][strat]
    P.VARIANTS[variant][strat] = prompt.split("\nReply after thinking")[0]
    try:
        if strategy == "full":
            people, calls = S.run_full(args, img, variant, think)
        else:
            people, calls = S.run_crops(args, img, variant, think, {})
    finally:
        P.VARIANTS[variant][strat] = orig
    errors = [c["error"] for c in calls if "error" in c]
    if errors:
        raise gr.Error(errors[0][:400])
    rows = [
        [i, *(p[k] for k in P.ITEMS), p.get("appearance") or " / ".join(
            str(p[f]) for f in ("head_seen", "eyes_seen", "torso_seen") if f in p)]
        for i, p in enumerate(people or [], 1)
    ]
    secs = sum(c.get("latency", 0) for c in calls)
    toks = sum(c.get("tokens") or 0 for c in calls)
    trunc = sum(c.get("finish") == "length" for c in calls)
    status = (f"{len(people or [])} people · {len(calls)} call(s) · {secs:.1f}s · {toks} tokens"
              + (f" · **{trunc} truncated at the token cap**" if trunc else "")
              + ("" if people is not None else " · **answer did not parse**"))
    raw = "\n\n".join(f"### {c['stage']}\n```\n{c.get('answer', '')}\n```" for c in calls)
    thinking = "\n\n".join(f"### {c['stage']}\n{c['thinking']}" for c in calls if c.get("thinking"))
    return draw(img, people or []), rows, status, raw, thinking or "(thinking off)"


# ── Review labels ────────────────────────────────────────────────────────────

FIELDS = [(k, a) for k in P.ITEMS for a in ("yes", "no", "na")]


def show_label(name: str | None):
    if not name:
        return [None, "", 0, False, False, ""] + [0] * len(FIELDS) + [""]
    e = load_labels().get(name, {})
    t = e.get("truth") or {}
    vals = [t.get(k, [0, 0, 0])[("yes", "no", "na").index(a)] for k, a in FIELDS]
    coco = ", ".join(f"{k}: {v}" for k, v in sorted(e.get("coco_counts", {}).items()))
    info = (f"{e.get('width')}×{e.get('height')} · {e.get('split')} · "
            f"{'**verified**' if e.get('verified') else 'draft by ' + e['draft_by'] if e.get('draft_by') else 'unlabelled'}"
            f"\n\nOriginal dataset labels (unreliable): {coco or '—'}")
    return ([str(DATA / "images" / name), info, t.get("people", 0), t.get("people_exact", True), e.get("verified", False),
             t.get("notes", "")] + vals + [""])


def save_label(name, people, exact, verified, notes, *vals):
    if not name:
        raise gr.Error("No image selected.")
    people = int(people)
    truth = {"people": people, "people_exact": bool(exact), "notes": notes}
    for j, item in enumerate(P.ITEMS):
        arr = [int(v) for v in vals[3 * j: 3 * j + 3]]
        if sum(arr) != people:
            raise gr.Error(f"{item}: yes+no+can't-tell = {sum(arr)}, but people = {people}.")
        truth[item] = arr
    L = load_labels()
    L[name]["truth"] = truth
    L[name]["verified"] = bool(verified)
    LABELS_PATH.write_text(json.dumps(L, indent=1))
    n_ver = sum(e.get("verified", False) for e in L.values())
    return f"Saved {name} ({'verified' if verified else 'draft'}). {n_ver} verified in total."


def step(name: str | None, delta: int):
    names = labelled_names()
    i = names.index(name) if name in names else -1
    return gr.update(value=names[(i + delta) % len(names)])


# ── Reports ──────────────────────────────────────────────────────────────────


def sweep_dirs() -> list[str]:
    root = DATA / "sweeps"
    return sorted((str(p.parent.name) for p in root.glob("*/report.md")), reverse=True) if root.exists() else []


def show_report(name: str | None) -> str:
    return (DATA / "sweeps" / name / "report.md").read_text() if name else ""


# ── UI ───────────────────────────────────────────────────────────────────────


def build() -> gr.Blocks:
    names = labelled_names()
    with gr.Blocks(title="Cosmos3-Edge PPE lab") as demo:
        gr.Markdown(f"## Cosmos3-Edge PPE lab\n`{ARGS.model}` at `{ARGS.endpoint}` · data `{DATA}`")
        with gr.Tab("Try a wording"):
            with gr.Row():
                with gr.Column(scale=1):
                    pick = gr.Dropdown(names, value=None, label="Dataset image (drafts first)")
                    image = gr.Image(label="Image (or upload / webcam)", type="filepath",
                                     sources=["upload", "webcam", "clipboard"])
                    with gr.Row():
                        strategy = gr.Radio(["full", "crops"], value="crops", label="Strategy")
                        think = gr.Checkbox(label="Thinking", value=False)
                    variant = gr.Dropdown(list(P.VARIANTS), value="V0_bool", label="Wording variant")
                    prompt = gr.Textbox(label="Prompt (editable; answer schema follows the variant)", lines=10,
                                        value=default_prompt("V0_bool", "crops", False))
                    with gr.Accordion("Limits", open=False):
                        max_nothink = gr.Number(512, label="Max tokens (thinking off)", precision=0)
                        max_think = gr.Number(4096, label="Max tokens (thinking on)", precision=0)
                        crop_min = gr.Number(768, label="Crop long side at least (px)", precision=0)
                    go = gr.Button("Ask", variant="primary")
                with gr.Column(scale=1):
                    annotated = gr.Image(label="Answer (crops: box per person; H V G = hat / vest / glasses, "
                                         "green yes · red no · amber unclear)", type="pil")
                    status = gr.Markdown()
                    table = gr.Dataframe(headers=["#", "hard_hat", "hi_vis_vest", "safety_glasses", "seen"],
                                         label="Per person", wrap=True)
                    with gr.Accordion("Raw answer", open=False):
                        raw = gr.Markdown()
                    with gr.Accordion("Thinking", open=False):
                        thinking = gr.Markdown()
            pick.change(pick_image, pick, image)
            for c in (variant, strategy, think):
                c.change(default_prompt, [variant, strategy, think], prompt)
            go.click(run, [image, strategy, variant, think, prompt, max_nothink, max_think, crop_min],
                     [annotated, table, status, raw, thinking], concurrency_limit=1)

        with gr.Tab("Review labels"):
            gr.Markdown("Counts are per image: of the people counted, how many **wear** the item, how many "
                        "**don't**, and how many you **can't tell**. Each row must add up to the people count. "
                        "Strict glasses rule: ordinary prescription glasses and sunglasses are *don't*.")
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        prev_btn = gr.Button("◀ Prev", size="sm")
                        lname = gr.Dropdown(names, value=names[0] if names else None, label="Image", scale=4)
                        next_btn = gr.Button("Next ▶", size="sm")
                    limg = gr.Image(type="filepath", label=None, height=620)
                    linfo = gr.Markdown()
                with gr.Column(scale=2):
                    people = gr.Number(label="People", precision=0)
                    exact = gr.Checkbox(label="People count is exact (score it)")
                    vals = []
                    for item in P.ITEMS:
                        with gr.Row():
                            for a in ("yes", "no", "na"):
                                vals.append(gr.Number(label=f"{item} {a if a != 'na' else 'can’t tell'}", precision=0))
                    notes = gr.Textbox(label="Notes", lines=2)
                    verified = gr.Checkbox(label="Verified by a human")
                    save = gr.Button("Save", variant="primary")
                    saved = gr.Markdown()
            outs = [limg, linfo, people, exact, verified, notes, *vals, saved]
            lname.change(show_label, lname, outs)
            prev_btn.click(lambda n: step(n, -1), lname, lname)
            next_btn.click(lambda n: step(n, 1), lname, lname)
            save.click(save_label, [lname, people, exact, verified, notes, *vals], saved)
            demo.load(show_label, lname, outs)

        with gr.Tab("Sweep reports"):
            with gr.Row():
                rep = gr.Dropdown(sweep_dirs(), label="Sweep", scale=4)
                refresh = gr.Button("Refresh", size="sm", scale=1)
            report = gr.Markdown()
            rep.change(show_report, rep, report)
            refresh.click(lambda: gr.update(choices=sweep_dirs()), None, rep)
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(
        server_name=ARGS.host, server_port=ARGS.port, allowed_paths=[str(DATA)],
        auth=tuple(ARGS.auth.split(":", 1)) if ARGS.auth else None,
    )
