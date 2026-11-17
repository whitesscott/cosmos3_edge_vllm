#!/usr/bin/env python3
"""Gradio frontend for the Cosmos3-Edge reasoner served by vLLM.

Tabs:
  Ask        any image + question; streamed answer with thinking shown separately;
             thinking on/off, sampling, optional JSON schema; bbox_2d boxes in the
             answer are drawn on the image.
  PPE check  the sweep's best pipeline (detect people in the native box format →
             drop phantom boxes → crop each person → yes/no per item, thinking off).

Start the server first (or let serve_cosmos3_edge_vllm.sh --gradio start both):
    ./serve_cosmos3_edge_vllm.sh --gradio
    ~/.git/uv/cosmos-cu134/bin/python cosmos3_vllm_app.py [--endpoint http://127.0.0.1:8000/v1]
"""

import argparse
import json
import time
import types
import urllib.error
import urllib.request

import gradio as gr
from PIL import Image, ImageDraw

import ppe_prompts as P
import ppe_sweep as S

COLORS = {"yes": (40, 200, 40), "no": (230, 40, 40), "unclear": (240, 180, 0)}
EXAMPLES = [
    "Describe what is happening in this image.",
    "Detect every person in this image. Output their bounding boxes in JSON.",
    "List every person and whether each wears a hard hat, a high-visibility vest, and safety glasses.",
    "Is anyone in this image in an unsafe position? Explain briefly.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="nvidia/Cosmos3-Edge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--auth", default=None, help="user:password")
    return p.parse_args()


ARGS = parse_args()


# ── Ask (streaming) ──────────────────────────────────────────────────────────


def stream_chat(body: dict):
    """Yield content deltas from vLLM's streaming chat completion; last item is the final usage dict."""
    req = urllib.request.Request(
        ARGS.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise gr.Error(f"vLLM {e.code}: {e.read().decode()[:400]}")
    except urllib.error.URLError as e:
        raise gr.Error(f"Cannot reach vLLM at {ARGS.endpoint}: {e.reason}")
    usage, finish = {}, None
    with resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices", []):
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta", {})
                text = (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if text:
                    yield ("think", text)
                if delta.get("content"):
                    yield ("text", delta["content"])
    yield ("done", {"usage": usage, "finish": finish})


def draw_boxes(img: Image.Image, answer: str) -> Image.Image | None:
    boxes = P.parse_boxes(P.extract_json(answer))
    if not boxes:
        return None
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.width // 300)
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = S.to_pixels(b, img.width, img.height, "1000")
        d.rectangle([x1, y1, x2, y2], outline=(0, 160, 255), width=lw)
        d.text((x1 + lw, y1 + lw), f"#{i}", fill=(255, 255, 255))
    return out


def max_model_len() -> int | None:
    """Context length the server was started with (reported by /v1/models)."""
    try:
        with urllib.request.urlopen(ARGS.endpoint.rstrip("/") + "/models", timeout=5) as r:
            return json.loads(r.read())["data"][0].get("max_model_len")
    except (OSError, KeyError, IndexError, ValueError):
        return None


def ask(image_path, question, think, max_tokens, temperature, schema_text, max_mp):
    if not question or not question.strip():
        raise gr.Error("Type a question.")
    content: list[dict] = []
    img = None
    if image_path:
        img = Image.open(image_path).convert("RGB")
        url = S.image_data_url(img, max_pixels=int(float(max_mp) * 1e6))
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": question})
    body: dict = {
        "model": ARGS.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    if think:
        # Greedy decoding + thinking loops to the token cap on this model.
        body.update(S.THINK_SAMPLING)
        if temperature:
            body["temperature"] = float(temperature)
    else:
        body["temperature"] = float(temperature or 0.0)
    if schema_text and schema_text.strip():
        if think:
            raise gr.Error("A JSON schema constrains the whole output, thinking included; turn thinking off.")
        try:
            schema = json.loads(schema_text)
        except json.JSONDecodeError as e:
            raise gr.Error(f"Schema is not valid JSON: {e}")
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}

    raw, thinking, t0, first = "", "", time.time(), None
    final: dict = {}
    for kind, val in stream_chat(body):
        if isinstance(val, dict):
            final = val
            break
        first = first or time.time()
        if kind == "think":
            thinking += val
        else:
            raw += val
        th, answer = P.strip_thinking(raw) if think else ("", raw)
        yield answer, (thinking + th) or "…", f"generating… {time.time() - t0:.1f}s", gr.update()
    th, answer = P.strip_thinking(raw) if think else ("", raw.strip())
    thinking = thinking + th
    usage, finish = final.get("usage") or {}, final.get("finish")
    prompt_toks, out_toks = usage.get("prompt_tokens"), usage.get("completion_tokens")
    stats = (f"{time.time() - t0:.1f}s · first token {((first or time.time()) - t0):.2f}s · "
             f"prompt {prompt_toks or '?'} tokens · answer {out_toks or '?'} tokens")
    if finish == "length":
        ctx = max_model_len()
        if ctx and prompt_toks and out_toks is not None and out_toks < int(max_tokens):
            # vLLM clamps max_tokens to what is left of the context instead of erroring.
            stats += (f" · **context full**: the prompt (mostly the image) used {prompt_toks} of {ctx} "
                      f"tokens, leaving {ctx - prompt_toks} for the answer. Lower *Max image megapixels*.")
        else:
            stats += " · **stopped at the Max tokens limit**"
    boxed = draw_boxes(img, answer) if img is not None else None
    yield answer, thinking or "(thinking off)", stats, gr.update(value=boxed, visible=boxed is not None)


# ── PPE check ────────────────────────────────────────────────────────────────


def ppe_check(image_path, crop_min_side):
    if not image_path:
        raise gr.Error("Upload an image first.")
    img = Image.open(image_path).convert("RGB")
    args = types.SimpleNamespace(
        model=ARGS.model, endpoint=ARGS.endpoint, timeout=300, box_scale="1000",
        max_tokens_nothink=512, max_tokens_think=4096, crop_margin=0.15, crop_min_side=int(crop_min_side),
    )
    t0 = time.time()
    people, calls = S.run_crops(args, img, "V0_bool", False, {})
    errors = [c["error"] for c in calls if "error" in c]
    if errors:
        raise gr.Error(errors[0][:400])
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.width // 300)
    rows = []
    for i, p in enumerate(people or [], 1):
        missing = [k for k in P.ITEMS if p[k] == "no"]
        d.rectangle(p["bbox_px"], outline=COLORS["no" if missing else "yes"], width=lw)
        d.text((p["bbox_px"][0] + lw, p["bbox_px"][1] + lw), f"#{i}", fill=(255, 255, 255))
        rows.append([i, *(p[k] for k in P.ITEMS), ", ".join(missing) or "—"])
    det = calls[0] if calls and calls[0].get("stage") == "detect" else {}
    dropped = len(det.get("boxes_raw") or []) - len(det.get("boxes") or [])
    alarms = sum(1 for r in rows if r[-1] != "—")
    status = (f"**{len(rows)} people · {alarms} with missing gear** · {len(calls)} calls · "
              f"{time.time() - t0:.1f}s" + (f" · {dropped} phantom box(es) dropped" if dropped > 0 else ""))
    return out, rows, status


# ── UI ───────────────────────────────────────────────────────────────────────


def build() -> gr.Blocks:
    with gr.Blocks(title="Cosmos3-Edge (vLLM)") as demo:
        gr.Markdown(f"## Cosmos3-Edge reasoner\n`{ARGS.model}` at `{ARGS.endpoint}`")
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="Image (optional)", type="filepath",
                                     sources=["upload", "webcam", "clipboard"])
                    question = gr.Textbox(label="Question", lines=4, value=EXAMPLES[0])
                    gr.Examples(EXAMPLES, question, label="Examples")
                    with gr.Row():
                        think = gr.Checkbox(label="Thinking", value=False)
                        max_tokens = gr.Number(1024, label="Max tokens", precision=0)
                        temperature = gr.Number(0, label="Temperature (0 = default)")
                    max_mp = gr.Slider(0.5, 16, value=4, step=0.5, label="Max image megapixels "
                                       "(≈1k tokens per MP; the server context is 16k tokens)")
                    with gr.Accordion("JSON schema (thinking off)", open=False):
                        schema = gr.Code(language="json", value="", label="Leave empty for free text")
                    go = gr.Button("Ask", variant="primary")
                with gr.Column(scale=1):
                    stats = gr.Markdown()
                    answer = gr.Markdown(label="Answer")
                    boxed = gr.Image(label="Boxes in the answer", visible=False, type="pil")
                    with gr.Accordion("Thinking", open=False):
                        thinking = gr.Markdown()
            go.click(ask, [image, question, think, max_tokens, temperature, schema, max_mp],
                     [answer, thinking, stats, boxed], concurrency_limit=2)
            question.submit(ask, [image, question, think, max_tokens, temperature, schema, max_mp],
                            [answer, thinking, stats, boxed], concurrency_limit=2)

        with gr.Tab("PPE check"):
            gr.Markdown("Best pipeline from the wording sweep: find people (native boxes, phantom boxes dropped) "
                        "→ crop each person → yes/no for hard hat, hi-vis vest, safety glasses. "
                        "Green box = all gear, red = something missing.")
            with gr.Row():
                with gr.Column(scale=1):
                    pimage = gr.Image(label="Image", type="filepath", sources=["upload", "webcam", "clipboard"])
                    crop_min = gr.Slider(448, 1024, value=768, step=64, label="Crop long side at least (px)")
                    pgo = gr.Button("Check PPE", variant="primary")
                with gr.Column(scale=1):
                    pout = gr.Image(label="Result", type="pil")
                    pstatus = gr.Markdown()
                    ptable = gr.Dataframe(headers=["#", "hard_hat", "hi_vis_vest", "safety_glasses", "missing"],
                                          label="Per person")
            pgo.click(ppe_check, [pimage, crop_min], [pout, ptable, pstatus], concurrency_limit=1)
    return demo


if __name__ == "__main__":
    auth = tuple(ARGS.auth.split(":", 1)) if ARGS.auth else None
    build().queue(default_concurrency_limit=2).launch(server_name=ARGS.host, server_port=ARGS.port, auth=auth)
