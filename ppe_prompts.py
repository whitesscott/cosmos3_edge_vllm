"""Question wordings, JSON schemas and answer parsing for the Cosmos3-Edge PPE sweep.

Each variant is one way of asking the same question: who is in the picture and
which of hard hat / hi-vis vest / safety glasses each person wears. Variants only
change wording and answer shape; strategy (full frame vs per-person crops) and
thinking on/off are separate sweep axes, so every variant works in every mode.

Answers are normalized to "yes" / "no" / "unclear" per person per item.
"""

import json
import re

ITEMS = ("hard_hat", "hi_vis_vest", "safety_glasses")

# What each item looks like, and what it is commonly confused with.
DEFINITIONS = {
    "hard_hat": "a rigid protective helmet (hard hat) worn on the head. Caps, hoods, beanies and "
    "hair do not count.",
    "hi_vis_vest": "a high-visibility safety vest or jacket: fluorescent yellow-green or orange fabric "
    "with reflective silver stripes, worn on the torso. Ordinary colored shirts, life jackets "
    "without hi-vis fabric, and harness straps alone do not count.",
    "safety_glasses": "protective eyewear worn over the eyes: clear or tinted safety glasses, often "
    "wraparound with side shields, or goggles. Ordinary prescription glasses, fashion "
    "sunglasses, and glasses that are not on the eyes (in a pocket, on the helmet, in a hand) "
    "do not count.",
}

TRISTATE = {"type": "string", "enum": ["yes", "no", "unclear"]}


def _person_schema(fields: dict) -> dict:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def _people_schema(person: dict) -> dict:
    return {
        "type": "object",
        "properties": {"people": {"type": "array", "items": person}},
        "required": ["people"],
        "additionalProperties": False,
    }


_BBOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}

# ── Variants ─────────────────────────────────────────────────────────────────
# person_fields: JSON object fields per person, in generation order (order matters:
# fields written first condition the later ones).


def _defs_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())


VARIANTS = {
    # Forum-style baseline: booleans, no definitions, no way to say "can't see".
    "V0_bool": {
        "full": "List every person in this image. For each person, report whether they are wearing a hard "
        "hat, a high-visibility vest, and safety glasses. Answer in JSON.",
        "crop": "This image is a close-up of one person. Report whether this person is wearing a hard hat, "
        "a high-visibility vest, and safety glasses. Answer in JSON.",
        "person_fields": {k: {"type": "boolean"} for k in ITEMS},
    },
    # Same, but the model may say it cannot see the item.
    "V1_tristate": {
        "full": "List every person in this image. For each person, answer yes, no, or unclear for each item: "
        "hard_hat, hi_vis_vest, safety_glasses. Use unclear when that part of the person is hidden, "
        "turned away, or too small to see. Answer in JSON.",
        "crop": "This image is a close-up of one person. Answer yes, no, or unclear for each item: hard_hat, "
        "hi_vis_vest, safety_glasses. Use unclear when that part of the person is hidden, turned away, "
        "or too small to see. Answer in JSON.",
        "person_fields": {k: TRISTATE for k in ITEMS},
    },
    # Tristate + what each item looks like and what does not count.
    "V2_definitions": {
        "full": "List every person in this image. For each person, answer yes, no, or unclear for each item, "
        "using these definitions:\n" + _defs_block() + "\nJudge only what is visible on that person; do not "
        "assume gear because the scene is a work site. Use unclear when that body part is hidden, turned "
        "away, or too small to see. Answer in JSON.",
        "crop": "This image is a close-up of one person. Answer yes, no, or unclear for each item, using these "
        "definitions:\n" + _defs_block() + "\nJudge only what is visible on this person; do not assume gear "
        "because the scene is a work site. Use unclear when that body part is hidden, turned away, or too "
        "small to see. Answer in JSON.",
        "person_fields": {k: TRISTATE for k in ITEMS},
    },
    # Definitions + a short appearance description written before the judgments.
    "V3_describe_first": {
        "full": "List every person in this image. For each person, first write a short appearance description "
        "(position in the image, what is on their head, over their eyes, and on their torso), then answer "
        "yes, no, or unclear for each item, using these definitions:\n" + _defs_block() + "\nJudge only what "
        "is visible on that person. Use unclear when that body part is hidden or too small. Answer in JSON.",
        "crop": "This image is a close-up of one person. First write a short description of what is on their "
        "head, over their eyes, and on their torso, then answer yes, no, or unclear for each item, using "
        "these definitions:\n" + _defs_block() + "\nJudge only what is visible. Use unclear when that body "
        "part is hidden or too small. Answer in JSON.",
        "person_fields": {"appearance": {"type": "string"}, **{k: TRISTATE for k in ITEMS}},
    },
    # Look at one body region at a time, write what is there, then judge that item.
    "V4_region_evidence": {
        "full": "List every person in this image. For each person, look at one body region at a time and "
        "write what you see there before judging it: the head (hard_hat), the eyes (safety_glasses), the "
        "torso (hi_vis_vest). Definitions:\n" + _defs_block() + "\nAnswer yes, no, or unclear for each item. "
        "Use unclear when that region is hidden, turned away, or too small to see. Answer in JSON.",
        "crop": "This image is a close-up of one person. Look at one body region at a time and write what you "
        "see there before judging it: the head (hard_hat), the eyes (safety_glasses), the torso "
        "(hi_vis_vest). Definitions:\n" + _defs_block() + "\nAnswer yes, no, or unclear for each item. Use "
        "unclear when that region is hidden, turned away, or too small to see. Answer in JSON.",
        "person_fields": {
            "head_seen": {"type": "string"},
            "hard_hat": TRISTATE,
            "eyes_seen": {"type": "string"},
            "safety_glasses": TRISTATE,
            "torso_seen": {"type": "string"},
            "hi_vis_vest": TRISTATE,
        },
    },
}

# Two-step: a free-text look first (no schema), then the V2 question with that text in
# the prompt. With thinking off, string fields inside a strict JSON schema come out as
# junk (", ") on Cosmos3-Edge, so in-schema "describe first" (V3/V4) never really
# describes; this variant gets an actual description in front of the judgment.
VARIANTS["V5_two_step"] = {
    **VARIANTS["V2_definitions"],
    "describe_full": "Describe every person in this image, one line each: where they are in the image, what "
    "is on their head, what is over their eyes, and what they wear on their torso. Mention colors, "
    "reflective stripes, and the exact type of any eyewear. Say 'not visible' when a body part is hidden, "
    "turned away, or too small to see.",
    "describe_crop": "Describe this person in three short lines: what is on their head, what is over their "
    "eyes, and what they wear on their torso. Mention colors, reflective stripes, and the exact type of any "
    "eyewear. Say 'not visible' when a body part is hidden, turned away, or too small to see.",
}


def two_step_prompt(variant: str, strategy: str, description: str, *, with_schema_hint: bool) -> str:
    text = (
        prompt_for(variant, strategy, with_schema_hint=with_schema_hint)
        + "\n\nNotes from a first look at this image (use them, but trust the image):\n"
        + description.strip()
    )
    if strategy == "full":
        # Under a strict schema an empty "people" list is the cheapest valid answer.
        text += ("\n\nReturn exactly one entry per person in the notes, in the same order, including "
                 "people whose body parts are not visible (answer unclear for those).")
    return text


# Stage 1 of the crops strategy: find people. Asked in the model's native grounding
# format ([{"bbox_2d": [x1,y1,x2,y2], "label": ...}], 0-1000 normalized) with no schema:
# forcing a strict "bbox" schema produced edge slivers, nested head boxes, runaway lists
# and out-of-range values; the native free-form answer was clean.
DETECT_PROMPT = (
    "Detect every person in this image, including small, distant, and partly hidden people. "
    "Output their bounding boxes in JSON."
)
DETECT_SCHEMA = None

# Grounded full-frame variant: each person entry starts with its native box, which
# ties every entry to a real person (V0-V5 invent a duplicate on most one-person images).
VARIANTS["V6_grounded"] = {
    "full": "Find every person in this image. For each person, give their bounding box (bbox_2d, "
    "normalized 0-1000) and answer yes, no, or unclear for each item, using these definitions:\n"
    + _defs_block() + "\nList each person once. Judge only what is visible on that person; do not assume "
    "gear because the scene is a work site. Use unclear when that body part is hidden, turned away, or too "
    "small to see. Answer in JSON.",
    "crop": VARIANTS["V2_definitions"]["crop"],
    "person_fields": {"bbox_2d": _BBOX, **{k: TRISTATE for k in ITEMS}},
    "crop_fields": {k: TRISTATE for k in ITEMS},
}


def filter_boxes(boxes: list[list[float]], *, min_side: float = 15, contain: float = 0.8,
                 edge: float = 990) -> list[list[float]]:
    """Drop phantom boxes (0-1000 units): out-of-range / degenerate, thin slivers along the
    image edge, near-duplicates, and boxes mostly inside a larger box (a head inside its body)."""
    clean = []
    for b in boxes:
        # Clamp rather than reject: the model sometimes garbles 999 into 9999 on a real person.
        b = [max(0.0, min(1000.0, v)) for v in b]
        if b[2] - b[0] < 3 or b[3] - b[1] < 3:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        at_edge = b[0] <= 1000 - edge or b[2] >= edge
        if at_edge and w < min_side * 4 and h < 300:  # sliver hugging the left/right border
            continue
        clean.append(b)

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def inter(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    clean.sort(key=area, reverse=True)
    kept: list[list[float]] = []
    for b in clean:
        if any(inter(b, k) >= contain * area(b) for k in kept):
            continue
        kept.append(b)
    return kept


def schema_for(variant: str, strategy: str) -> dict:
    v = VARIANTS[variant]
    if strategy == "crop":
        return _person_schema(v.get("crop_fields", v["person_fields"]))
    return _people_schema(_person_schema(v["person_fields"]))


def prompt_for(variant: str, strategy: str, *, with_schema_hint: bool) -> str:
    """Prompt text. Without structured outputs (thinking on), spell out the JSON shape."""
    v = VARIANTS[variant]
    text = v["crop" if strategy == "crop" else "full"]
    if with_schema_hint:
        fields = v.get("crop_fields", v["person_fields"]) if strategy == "crop" else v["person_fields"]
        shape = {k: _hint(s) for k, s in fields.items()}
        example = shape if strategy == "crop" else {"people": [shape]}
        text += "\nReply after thinking with only this JSON shape: " + json.dumps(example)
    return text


def detect_prompt() -> str:
    return DETECT_PROMPT


def _hint(s: dict) -> str:
    if "enum" in s:
        return "|".join(s["enum"])
    if s.get("type") == "array":
        return "[x1, y1, x2, y2]"
    return {"boolean": "true|false", "string": "..."}.get(s.get("type", ""), "...")


# ── Parsing ──────────────────────────────────────────────────────────────────


def strip_thinking(text: str) -> tuple[str, str]:
    """Split raw output into (thinking, answer). The template opens <think> in the prompt,
    so the completion holds '...reasoning...</think>answer' (or no tag when thinking is off)."""
    if "</think>" in text:
        think, answer = text.rsplit("</think>", 1)
        return think.replace("<think>", "").strip(), answer.strip()
    if "<think>" in text:  # opened but never closed: truncated mid-thought
        return text.split("<think>", 1)[1].strip(), ""
    return "", text.strip()


def extract_json(text: str):
    """First JSON object (or array) in text; None if there is none."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue
    return None


def norm_answer(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if s in ("yes", "true", "y", "1", "wearing", "present"):
        return "yes"
    if s in ("no", "false", "n", "0", "not wearing", "absent", "missing"):
        return "no"
    return "unclear"


def parse_people(obj) -> list[dict] | None:
    """Normalize a full-frame answer to [{item: yes|no|unclear, ...}, ...]."""
    if isinstance(obj, dict):
        obj = obj.get("people", obj.get("persons", obj.get("workers")))
    if not isinstance(obj, list):
        return None
    return [parse_person(p) for p in obj if isinstance(p, dict)]


def parse_person(obj: dict) -> dict:
    out = {k: norm_answer(obj.get(k, "unclear")) for k in ITEMS}
    for k in ("appearance", "head_seen", "eyes_seen", "torso_seen", "bbox", "bbox_2d"):
        if k in obj:
            out[k] = obj[k]
    return out


def parse_boxes(obj) -> list[list[float]] | None:
    """[[x1,y1,x2,y2], ...] from a detection answer (accepts a few common shapes)."""
    if isinstance(obj, dict):
        if "bbox_2d" in obj or "bbox" in obj:
            obj = [obj]
        else:
            obj = obj.get("people", obj.get("persons", obj.get("boxes", obj.get("detections"))))
    if not isinstance(obj, list):
        return None
    boxes = []
    for p in obj:
        b = p.get("bbox", p.get("bbox_2d", p.get("box"))) if isinstance(p, dict) else p
        if isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
            boxes.append([float(x) for x in b])
    return boxes
