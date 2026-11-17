#!/usr/bin/env python3
"""Gradio front end for cosmos-framework (Cosmos3 omni generation), no Ray.

The model is loaded once in this process via cosmos_framework's OmniInference
(the same path `python -m cosmos_framework.scripts.inference` uses) and every
request runs on that resident pipeline. Requests are queued one at a time.

Fields left blank fall back to cosmos_framework's per-mode defaults
(cosmos_framework/inference/defaults/<mode>/sample_args.json + model-specific
overrides, e.g. 121 frames for Cosmos3-Edge).
"""

# init_script() must run before torch / cosmos model imports (same as scripts/inference.py).
from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

import argparse
import json
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
from cosmos_framework.utils import log

DEFAULT_CHECKPOINT = str(
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
)
DEFAULT_REPO = Path.home() / ".git/cosmos/cosmos-framework"

MODES = {
    "Text → Image": "text2image",
    "Text → Video": "text2video",
    "Image → Video": "image2video",
    "Video → Video": "video2video",
}
# Example inputs shipped with the repo, keyed by mode.
EXAMPLE_FILES = {
    "text2image": "t2i.json",
    "text2video": "t2v.json",
    "image2video": "i2v.json",
    "video2video": "v2v.json",
}
RESOLUTIONS = ["(default)", "256", "480"]  # Cosmos3-Edge supports 256 / 480 (inference/args.py)
ASPECT_RATIOS = ["(default)", "16,9", "9,16", "1,1", "4,3", "3,4"]
MODE_LABELS = {v: k for k, v in MODES.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov"}

PIPE = None
PIPE_LOCK = threading.Lock()
ARGS: argparse.Namespace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT, help="Local checkpoint dir or HF name")
    p.add_argument("--output-dir", default=str(Path.home() / "cosmos-out/gradio"), help="Where generations are saved")
    p.add_argument("--repo", default=str(DEFAULT_REPO), help="cosmos-framework checkout (for example inputs)")
    p.add_argument(
        "--library-root",
        default=str(Path.home() / "cosmos-out"),
        help="Folder browsed by the Library tab (includes CLI runs, not just this app's output)",
    )
    p.add_argument("--parallelism-preset", default="latency", choices=["latency", "throughput"])
    p.add_argument("--guardrails", action="store_true", help="Enable guardrails (downloads extra models)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Gradio built-in public link")
    p.add_argument("--auth", default=None, help="Require login, 'user:password' (recommended with ngrok/--share)")
    return p.parse_args()


def load_pipeline(args: argparse.Namespace):
    setup = OmniSetupOverrides.model_construct(
        diffusion_cache=True,  # same default as scripts/inference.py
        checkpoint_path=args.checkpoint_path,
        output_dir=Path(args.output_dir),
        parallelism_preset=args.parallelism_preset,
        guardrails=args.guardrails,
    )
    setup_args = setup.build_setup()
    init_output_dir(setup_args.output_dir)
    log.info(f"Loading {args.checkpoint_path} ...")
    t0 = time.time()
    pipe = setup_args.get_inference_cls().create(setup_args)
    log.success(f"Model loaded in {time.time() - t0:.1f}s")
    return pipe


def load_example(mode_label: str):
    """Return (prompt, vision_path) from the repo's example JSON for this mode."""
    mode = MODES[mode_label]
    path = Path(ARGS.repo) / "inputs/omni" / EXAMPLE_FILES[mode]
    try:
        data = json.loads(path.read_text())
    except OSError as e:
        raise gr.Error(f"Cannot read {path}: {e}")
    return data.get("prompt", ""), data.get("vision_path", "")


def update_mode(mode_label: str):
    mode = MODES[mode_label]
    is_video = mode != "text2image"
    return (
        gr.update(visible=mode == "image2video"),  # image upload
        gr.update(visible=mode == "video2video"),  # video upload
        gr.update(visible=mode in ("image2video", "video2video")),  # vision URL
        gr.update(visible=is_video),  # num_frames
        gr.update(visible=is_video),  # fps
    )


def _opt_int(v):
    return None if v in (None, "") or (isinstance(v, (int, float)) and v <= 0) else int(v)


def _opt_float(v):
    return None if v in (None, "") or (isinstance(v, (int, float)) and v <= 0) else float(v)


def generate(
    mode_label, prompt, negative_prompt, image_in, video_in, vision_url,
    seed, randomize_seed, num_steps, guidance, resolution, aspect_ratio, num_frames, fps,
):
    mode = MODES[mode_label]
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty (try 'Load example').")

    vision_path = None
    if mode == "image2video":
        vision_path = image_in or (vision_url or None)
        if not vision_path:
            raise gr.Error("Image → Video needs an input image (upload or URL).")
    elif mode == "video2video":
        vision_path = video_in or (vision_url or None)
        if not vision_path:
            raise gr.Error("Video → Video needs an input video (upload or URL).")

    seed = random.randint(0, 2**31 - 1) if randomize_seed or seed in (None, "") else int(seed)
    name = f"{mode}_{datetime.now():%Y%m%d_%H%M%S}_s{seed}"
    out_dir = Path(ARGS.output_dir) / name

    fields = dict(
        name=name,
        model_mode=mode,
        prompt=prompt,
        seed=seed,
        output_dir=out_dir,
        vision_path=vision_path,
        negative_prompt=negative_prompt or None,
        num_steps=_opt_int(num_steps),
        guidance=_opt_float(guidance),
        resolution=None if resolution == "(default)" else resolution,
        aspect_ratio=None if aspect_ratio == "(default)" else aspect_ratio,
    )
    if mode != "text2image":
        fields["num_frames"] = _opt_int(num_frames)
        fields["fps"] = _opt_int(fps)
    overrides = OmniSampleOverrides.model_validate({k: v for k, v in fields.items() if v is not None})
    overrides.download(out_dir / "inputs")  # fetch URL inputs (no-op for local uploads)

    if PIPE is None:
        raise gr.Error("Model is not loaded yet.")
    with PIPE_LOCK:
        t0 = time.time()
        sample_args = overrides.build_sample(model_config=PIPE.model_config)
        result = PIPE.generate([sample_args])[0]
        elapsed = time.time() - t0

    if result.status != "success":
        raise gr.Error(f"Generation {result.status}: {result.message}")
    files = [Path(f) for o in result.outputs for f in o.files]
    video = next((str(f) for f in files if f.suffix.lower() in (".mp4", ".webm", ".mov")), None)
    image = next((str(f) for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")), None)

    used = json.loads((out_dir / "sample_args.json").read_text()) if (out_dir / "sample_args.json").exists() else {}
    info = (
        f"**{mode}** in {elapsed:.1f}s · seed `{seed}` · "
        f"steps {used.get('num_steps', '?')} · guidance {used.get('guidance', '?')} · "
        f"res {used.get('resolution', '?')} ({used.get('aspect_ratio', '?')})"
        + (f" · {used.get('num_frames', '?')} frames @ {used.get('fps', '?')} fps" if mode != "text2image" else "")
        + f"\n\nSaved to `{out_dir}`"
    )
    return (
        gr.update(value=image, visible=image is not None),
        gr.update(value=video, visible=video is not None),
        info,
        seed,
    )


# ── Library (review generated files) ─────────────────────────────────────────


def scan_media(kind: str, query: str, limit: int) -> list[Path]:
    """Generated images/videos under the library root, newest first (skips downloaded inputs)."""
    root = Path(ARGS.library_root)
    exts = {"Videos": VIDEO_EXTS, "Images": IMAGE_EXTS}.get(kind, IMAGE_EXTS | VIDEO_EXTS)
    query = (query or "").lower()
    files = [
        f
        for f in root.rglob("*")
        if f.suffix.lower() in exts
        and "inputs" not in f.relative_to(root).parts
        and f.is_file()
        and (not query or query in str(f.relative_to(root)).lower())
    ]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[: int(limit)]


def refresh_library(kind: str, query: str, limit: int):
    files = scan_media(kind, query, limit)
    root = Path(ARGS.library_root)
    items = [(str(f), str(f.parent.relative_to(root))) for f in files]
    return items, [str(f) for f in files], f"{len(files)} file(s) in `{root}`"


def _sample_args_for(path: Path) -> dict:
    for d in (path.parent, path.parent.parent):
        f = d / "sample_args.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def describe(path_str: str | None):
    """Viewer updates for one file: image, video, details markdown, download, selected path."""
    if not path_str:
        return gr.update(visible=False), gr.update(visible=False), "", None, None
    path = Path(path_str)
    if not path.is_file():
        return gr.update(visible=False), gr.update(visible=False), f"Not a file: `{path}`", None, None
    is_video = path.suffix.lower() in VIDEO_EXTS
    is_image = path.suffix.lower() in IMAGE_EXTS
    a = _sample_args_for(path)
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"`{path}`  ·  {path.stat().st_size / 1e6:.1f} MB  ·  {stamp}"]
    if a:
        lines.append(
            f"**{a.get('model_mode', '?')}** · seed `{a.get('seed')}` · steps {a.get('num_steps')} · "
            f"guidance {a.get('guidance')} · res {a.get('resolution')} ({a.get('aspect_ratio')})"
            + (f" · {a.get('num_frames')} frames @ {a.get('fps')} fps" if is_video else "")
        )
        prompt = str(a.get("prompt") or "")
        if prompt:
            lines.append("**Prompt:** " + (prompt[:600] + " …" if len(prompt) > 600 else prompt))
    return (
        gr.update(value=str(path) if is_image else None, visible=is_image),
        gr.update(value=str(path) if is_video else None, visible=is_video),
        "\n\n".join(lines),
        str(path) if (is_image or is_video) else None,
        str(path),
    )


def on_gallery_select(paths: list[str], evt: gr.SelectData):
    return describe(paths[evt.index] if paths and evt.index < len(paths) else None)


def on_explorer_select(selection):
    if isinstance(selection, list):
        selection = selection[0] if selection else None
    return describe(selection)


def reuse_settings(path_str: str | None):
    """Copy a past run's mode / prompt / seed back into the Generate tab."""
    if not path_str:
        raise gr.Error("Select a file first.")
    a = _sample_args_for(Path(path_str))
    if not a:
        raise gr.Error("No sample_args.json found next to this file.")
    label = MODE_LABELS.get(a.get("model_mode"), "Text → Video")
    return gr.update(value=label), a.get("prompt", ""), a.get("seed", 0), gr.update(selected="generate")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Cosmos3 Studio") as demo:
        gr.Markdown(f"## Cosmos3 Studio\n`{ARGS.checkpoint_path}` · outputs in `{ARGS.output_dir}`")
        with gr.Tabs() as tabs:
            # ── Generate ─────────────────────────────────────────────────────
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mode = gr.Radio(list(MODES), value="Text → Video", label="Mode")
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=8,
                            max_lines=20,
                            placeholder="Describe the scene. Long, detailed prompts work best — "
                            "'Load example' shows the format.",
                        )
                        example_btn = gr.Button("Load example for this mode", size="sm")
                        image_in = gr.Image(label="Input image", type="filepath", visible=False)
                        video_in = gr.Video(label="Input video", visible=False)
                        vision_url = gr.Textbox(label="…or input URL", visible=False)
                        with gr.Accordion("Settings (blank / 0 = model default)", open=False):
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=0, precision=0)
                                randomize = gr.Checkbox(label="Randomize seed", value=False)
                            with gr.Row():
                                num_steps = gr.Number(label="Steps", value=None, precision=0)
                                guidance = gr.Number(label="Guidance", value=None)
                            with gr.Row():
                                resolution = gr.Dropdown(RESOLUTIONS, value="(default)", label="Resolution")
                                aspect = gr.Dropdown(ASPECT_RATIOS, value="(default)", label="Aspect ratio")
                            with gr.Row():
                                num_frames = gr.Number(label="Frames", value=None, precision=0)
                                fps = gr.Number(label="FPS", value=None, precision=0)
                            negative = gr.Textbox(label="Negative prompt (blank = model default)", lines=2)
                        run_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        out_image = gr.Image(label="Result", visible=False, type="filepath")
                        out_video = gr.Video(label="Result", visible=True, autoplay=True, loop=True)
                        info = gr.Markdown()
                        browse_btn = gr.Button("📂 Browse files (Library)", size="sm")

            # ── Library ──────────────────────────────────────────────────────
            with gr.Tab("Library", id="library"):
                with gr.Row():
                    kind = gr.Radio(["All", "Videos", "Images"], value="All", label="Show", scale=2)
                    query = gr.Textbox(label="Search path", placeholder="e.g. text2video, exp/, s42", scale=3)
                    limit = gr.Slider(6, 200, value=36, step=6, label="Max items", scale=2)
                    refresh_btn = gr.Button("Refresh", scale=1)
                lib_status = gr.Markdown()
                lib_paths = gr.State([])
                selected = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        gallery = gr.Gallery(
                            label="Newest first — click to open", columns=4, height=620, allow_preview=False
                        )
                        with gr.Accordion("📂 Browse folders", open=True):
                            explorer = gr.FileExplorer(
                                root_dir=ARGS.library_root,
                                glob="**/vision.*",
                                ignore_glob="**/*.meta",
                                file_count="single",
                                label=ARGS.library_root,
                                max_height=400,
                            )
                    with gr.Column(scale=2):
                        view_image = gr.Image(label="Selected", visible=False, type="filepath")
                        view_video = gr.Video(label="Selected", visible=False, autoplay=True, loop=True)
                        view_info = gr.Markdown()
                        with gr.Row():
                            download = gr.File(label="Download", scale=2)
                            reuse_btn = gr.Button("Reuse in Generate", scale=1)

        # Generate events
        mode.change(update_mode, mode, [image_in, video_in, vision_url, num_frames, fps])
        example_btn.click(load_example, mode, [prompt, vision_url])
        run_btn.click(
            generate,
            [mode, prompt, negative, image_in, video_in, vision_url,
             seed, randomize, num_steps, guidance, resolution, aspect, num_frames, fps],
            [out_image, out_video, info, seed],
            concurrency_limit=1,
        )

        # Library events
        lib_inputs = [kind, query, limit]
        lib_outputs = [gallery, lib_paths, lib_status]
        refresh_btn.click(refresh_library, lib_inputs, lib_outputs)
        kind.change(refresh_library, lib_inputs, lib_outputs)
        query.submit(refresh_library, lib_inputs, lib_outputs)
        tabs.select(refresh_library, lib_inputs, lib_outputs)  # picks up new generations on tab switch
        viewer = [view_image, view_video, view_info, download, selected]
        gallery.select(on_gallery_select, lib_paths, viewer)
        explorer.change(on_explorer_select, explorer, viewer)
        reuse_btn.click(reuse_settings, selected, [mode, prompt, seed, tabs])
        browse_btn.click(lambda: gr.update(selected="library"), None, tabs).then(
            refresh_library, lib_inputs, lib_outputs
        )
        demo.load(refresh_library, lib_inputs, lib_outputs)
    return demo


def main():
    global ARGS, PIPE
    ARGS = parse_args()
    PIPE = load_pipeline(ARGS)
    auth = tuple(ARGS.auth.split(":", 1)) if ARGS.auth else None
    build_ui().queue(default_concurrency_limit=1).launch(
        server_name=ARGS.host,
        server_port=ARGS.port,
        share=ARGS.share,
        auth=auth,
        allowed_paths=[ARGS.output_dir, ARGS.library_root],
    )


if __name__ == "__main__":
    main()
