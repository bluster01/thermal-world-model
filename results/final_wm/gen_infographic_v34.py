"""Nano Banana Pro: FinalWM architecture infographic v3/v4 -- REAL internals.

v3: mechanism drawn inside every card (visual motifs + short micro-labels).
v4: v3 + input/output annotations on the main arrows.
Facts extracted from src/final_wm/{model,transition,closure,boundary,observer}.py
"""
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

KEY = os.environ.get("OPENROUTER_API_KEY")
if not KEY:
    raise RuntimeError("OPENROUTER_API_KEY env var required (do not hardcode secrets)")
MODEL = "google/gemini-3-pro-image"
OUT = Path("/home/bluster/projectA/thermal-world-model/results/final_wm/infographic_v0")
OUT.mkdir(parents=True, exist_ok=True)

CORE = """Draw a complete editorial scientific infographic, wide 16:9, pure white background, explaining the REAL architecture of an auditable thermal world model for a supercritical power plant. Draw every mechanism visually inside each card; keep text labels short.

MAIN FLOW, left to right:

1. OBSERVED HISTORY card (left): a 16-minute window of plant data drawn as THREE thin horizontal time-series strips stacked vertically, tiny labels "5 temperatures", "2 valve positions", "7 boundary channels". THIS IS KEY: 96 steps of 10 seconds each.

2. PROBABILISTIC OBSERVER card (cyan): draw a small recurrent-cell motif (stacked arrows reading the strips) that condenses to one hidden vector, then splits into TWO output heads labeled "state mean" and "state variance". Add a tiny pressure gauge motif with two soft branches labeled "subcritical / supercritical" feeding the heads. The observer corrects a five-point anchored steady state; draw the anchor as a small dim dashed base block under the heads.

3. ONE SHARED PHYSICS-STATE TRANSITION card (center, large): THIS IS KEY. Draw THREE vertical superheater-stage columns, each stage showing a small closed loop: "steam inventory" box on top, "metal thermal capacity" box below, thin fin motif between them for metal-to-steam heat transfer. Incoming paths into the stages: (a) coal symbol flowing through a small delay coil labeled "fuel lag"; (b) a small valve symbol flowing into a droplet symbol then into the steam box, labeled "valve -> spray -> mixing lag"; (c) a thin pressure gauge labeled "pressure slope". A thin vertical dashed divider inside the card separates two identical halves labeled "prediction use" and "action rollout use" -- the SAME physics block, reused.

4. PROBABILISTIC TEMPERATURE ROLLOUT card (right, pale green): 18 future steps drawn as a fan of fading uncertainty bands with a mean line, tiny label "10 s x 18 = 180 s" and "5 temperatures, mean +/- uncertainty".

5. TOP PATH: LOGGED OR PROPOSED ACTIONS card (small) -> pale amber ACTION-SUPPORT GATE card drawn as a narrow corridor with a dashed box inside labeled "support box from history" and a tiny flag "out-of-support: flagged"; solid arrow from gate into the central transition.

6. BOTTOM PATH: DECLARED BOUNDARY SCENARIO card (small) -> pale teal BOUNDARY MODEL card: draw a GRU encoder cell feeding an autoregressive decoder cell that emits 18 future steps of 7 channels with thin uncertainty whiskers; solid arrow into the central transition.

7. BELOW CENTER: ACTION-BLIND RESIDUAL CLOSURE card (pale teal): draw a tiny 2-layer MLP (two small stacked boxes with tanh marks), inputs drawn as ONLY the current state and a whitelist of current boundary channels, output drawn as per-stage power-correction arrows ("+delta-P" into steam, "-delta-P" out of metal -- heat moved, not created). A padlock badge on the card.

8. ONE PROHIBITED PATH ONLY: a dashed RED arrow from a small gray chip "TRUE FUTURE BOUNDARY" pointing toward the BOUNDARY MODEL, crossed by one red X, tiny caption "forecast mode: REJECT".

STORY ELEMENTS:
- Title band: "An Auditable Thermal World Model"
- Subtitle: "One shared physics-state transition drives prediction and action rollout -- with explicit information permissions"
- 3-step strip: "1. Encode 16 min of history into a probabilistic plant state" / "2. Roll the same physics 180 s forward under a declared scenario" / "3. Simulate any logged or proposed valve action through the support gate"
- Legend row: navy = information flow, teal = scenario / action, red dashed = prohibited
- Footer tagline: "Every path has a permission."

TEXT RULES: render ONLY the labels listed above; no other words anywhere. No paragraphs inside cards.

STYLE: Nature-style editorial vector infographic; restrained navy outlines, soft cyan state modules, pale amber gate, pale teal closure and boundary, pale green output, red ONLY for the single prohibited path. Flat 2D rounded cards, thin precise non-overlapping connectors, generous whitespace, no gradients, no decorative icons, no 3D geometry, no axes, no invented mechanisms, no accuracy claims, no extra arrows."""

V3 = CORE

V4 = CORE.replace(
    "- Legend row: navy = information flow, teal = scenario / action, red dashed = prohibited",
    """- Legend row: navy = information flow, teal = scenario / action, red dashed = prohibited
- Tiny arrow annotations ONLY on the four main arrows (no other arrow labels): history-to-observer "96 x 14 channels"; observer-to-transition "plant state, mean & variance"; transition-to-rollout "5 temperatures"; closure-to-transition "per-stage power corrections""")

def gen(tag, prompt):
    body = json.dumps({
        "model": MODEL,
        "modalities": ["text", "image"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body)
    req.add_header("Authorization", f"Bearer {KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("HTTP-Referer", "http://localhost")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        if "error" in d:
            print(f"[{tag}] API error: {d['error']}")
            return
        images = d["choices"][0]["message"].get("images", [])
        for i, img in enumerate(images):
            url = img["image_url"]["url"]
            if url.startswith("data:"):
                b64 = url.split(",", 1)[1]
                path = OUT / (f"{tag}.png" if i == 0 else f"{tag}_{i}.png")
                path.write_bytes(base64.b64decode(b64))
                print(f"[{tag}] saved {path} ({path.stat().st_size} bytes) in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"[{tag}] FAILED: {e}")

gen("v3_internals", V3)
gen("v4_io_arrows", V4)
