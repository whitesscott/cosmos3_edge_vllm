# Cosmos3-Edge PPE question-wording lab

Find question wordings that make Cosmos 3 Edge read small protective gear (hard hats,
hi-vis vests, safety glasses) as reliably as it finds people. Background: the forum
write-up (Cosmos 3 Edge on Thor) found 44/45 people but kept only 13/15 real gear
alarms and raised 3 false ones.

## Files

| file | what |
|---|---|
| `serve_cosmos3_edge_vllm.sh` | vLLM OpenAI server for the reasoner (port 8000), venv `cosmos-cu134`. Thor fixes + FlashInfer shim pre-flight; `--gradio` / `--lab` also start a frontend |
| `cosmos3_vllm_app.py` | Gradio frontend (port 7860): ask anything (streamed, thinking toggle, JSON schema, boxes drawn) + one-click PPE check |
| `ppe_prompts.py` | wording variants V0–V5, JSON schemas, answer parsing |
| `ppe_sweep.py` | headless sweep: strategy × variant × thinking over labelled images → `report.md` |
| `ppe_lab.py` | Gradio lab (port 7861): try/edit wordings, review labels, read sweep reports |
| `extract_frames.py` | pull frames from an .mp4 (same method as physical-ai-safety-observability) |
| `prep_coco_ppe.py` | build an image pool + `labels.json` from a Roboflow COCO export |

Data lives in `~/ppe-data/construction_ppe/` (`images/`, `labels.json`, `sweeps/`).
Source: `emanueledecandia/construction-safety-object-detection` (CC BY 4.0, Roboflow
"Construction Site Safety"). Its own labels are not usable as truth (no glasses class,
people/vests not exhaustively labelled, mosaics, mirror-padded 416px images), so 43
clean images carry hand-drafted `truth` (`verified: false`, `draft_by: claude`).

## Headless (tonight)

```bash
./serve_cosmos3_edge_vllm.sh > /tmp/cosmos3_edge_vllm.log 2>&1 &   # ~65 s to ready
python3 ppe_sweep.py --include-drafts --thinking off               # all variants, full + crops
python3 ppe_sweep.py --include-drafts --thinking on --strategies full --variants V0_bool,V2_definitions
python3 ppe_sweep.py --dry-run                                     # show prompts + schemas
```

Each sweep writes `~/ppe-data/construction_ppe/sweeps/<name>/{results.jsonl,summary.json,report.md}`.

## With a monitor (tomorrow)

```bash
./serve_cosmos3_edge_vllm.sh --lab              # vLLM + ppe_lab.py, http://127.0.0.1:7861
./serve_cosmos3_edge_vllm.sh --gradio           # vLLM + cosmos3_vllm_app.py, http://127.0.0.1:7860
./serve_cosmos3_edge_vllm.sh --tunnel --auth user:pass   # + cloudflared (TUNNEL_TOKEN or ~/.config/cloudflared/gradio.token)
```

1. **Review labels**: step through the drafts (they come first), fix counts, tick
   *Verified*, Save. Each item row is wearing / not wearing / can't tell and must add up
   to the people count. Strict glasses rule: prescription glasses and sunglasses = not wearing.
2. **Try a wording**: pick an image (or webcam), strategy, variant, thinking; edit the
   prompt freely. Crops mode draws one box per person with H/V/G in green/red/amber.
3. Re-run the sweep without `--include-drafts` to score only verified labels.

## Scoring

Per image and item: *real alarm* = truth has someone not wearing it; *kept* = the model
also answered "no" for someone; *false alarm* = the model answered "no" on an image where
everyone truly wears it. Image-items with a "can't tell" person and no confirmed
violation are not scored. Same idea as the forum's gear alarms, at image level.

## Axes

- **strategy** `full` (one question) vs `crops` (detect people → crop with 15% margin,
  upscale to ≥768 px long side → ask about one person). Detection boxes are Qwen-style
  `bbox_2d` normalized 0–1000 (verified on ppe_0750).
- **thinking** off: strict JSON via xgrammar, 512 tokens, greedy. On: free text, JSON
  parsed after `</think>`, 4096 tokens, temperature 0.6 / top_p 0.95 / top_k 20
  (greedy + thinking loops on this model). ~25 s and ~1.2k tokens per call vs ~2 s.
- **variants**
  - V0_bool: forum-style booleans, no definitions
  - V1_tristate: yes / no / unclear
  - V2_definitions: tristate + what each item looks like and what does not count
    (life jackets, prescription glasses, glasses in a pocket, "it's a work site")
  - V3_describe_first / V4_region_evidence: description fields before the judgments
  - V5_two_step: free-text description call, then V2 with that description in the prompt
  - V6_grounded: V2 with a native `bbox_2d` first in every person entry (full frame);
    entries are filtered like detection boxes

## Results so far (2026-11-16, 43 draft-labelled images, thinking off)

Best: **crops + V0_bool** — native-format person detection (no schema) → `filter_boxes`
→ crop each person → plain boolean question. People exact on 33/33 exactly-counted
images; kept alarms hard hat 12/12, vest 23/24, glasses 17–18/18; false alarms 0/0/0–1.
Stable across crop margin 5–15% and crop size 448–1024 px.

What mattered, in order:
1. **Phantom people.** Full-frame answers invent extra people on most one-person images
   (V3: 21/21), and the phantom entries carry the false alarms. Fixed by detecting people
   in the model's native `bbox_2d` format without a schema and filtering edge slivers,
   nested head boxes and duplicates. Forcing a strict `bbox` schema caused the phantoms.
2. **One person per question.** Crops beat the full frame even without upscaling.
3. **No "unclear" option.** Tristate wordings (V1/V2/V6) hedge 30–55% of the time and
   lose recall; booleans are decisive and, once phantoms are gone, not trigger-happy.
4. Wording details (definitions, describe-first, two-step) mattered less than 1–3;
   two-step (V5) was clearly worse (people 61/102 full frame).
5. **Thinking is not needed.** Full-frame V0 with thinking on cuts false alarms
   13/3/5 → 2/0/2 (it suppresses phantoms) but keeps fewer vests (21/24), has parse
   failures, and takes ~6 s/image vs ~1.7 s for crops + V0 with thinking off.

| run (thinking off unless noted) | people | hat kept | vest kept | glasses kept | false hat/vest/glasses |
|---|---|---|---|---|---|
| full/V0_bool (forum-style) | 90/102 | 11/12 | 23/24 | 15/18 | 13/3/5 |
| full/V3_describe_first | 86/102 | 11/12 | 22/24 | 16/18 | 4/1/1 |
| full/V0_bool, thinking on | 90/102 | 11/12 | 21/24 | 15/18 | 2/0/2 |
| full/V6_grounded | 87/102 | 9/12 | 20/24 | 14/18 | 2/0/2 |
| crops/V0_bool, old schema detection | 88/102 | 9/12 | 20/24 | 13/18 | 19/8/5 |
| **crops/V0_bool, native detection + filter** | **92/102** | **12/12** | **23/24** | **17/18** | **0/0/0** |
| crops/V2_definitions, native detection | 92/102 | 6/12 | 20/24 | 8/18 | 2/0/0 |

Clean (no-violation) images available for false alarms: 28 hat / 13 vest / 5 glasses —
glasses false-alarm numbers rest on only 5 images.

Sweeps: `~/ppe-data/construction_ppe/sweeps/{think_off_v1,think_off_v5,think_on_full,crops_ablate_*}`
(think_off_v1 used the old schema detection for crops); the native-detection V0/V6 run
is in `sweeps/native_detect_v0_v6`.

Remaining misses: crop bleed (#117: neighbour's vest inside a no-vest worker's crop) and
one borderline glasses label (#43). Image numbers (#43, #117, …) map to file names in `~/ppe-data/construction_ppe/draft_index.json`. Labels are drafts — verify before trusting numbers.

## Known model/server behaviour

- With thinking off, **free-text string fields inside a strict schema come out as junk**
  (`", "`), so V3/V4 only change field order in that mode; V5 is the working
  describe-then-judge. A `pattern` constraint on strings is rejected by the server.
- Under a strict schema an empty `people` list is the cheapest answer; V5 asks for one
  entry per described person.
- V1 (bare tristate) answers "unclear" ~70% of the time in early runs.
