"""Nano Banana Pro: FinalWM architecture infographic, 2 variants (2026-08-25).

V1: label-light story version. V2: + narrative callouts.
Conceptual composition references only -- TikZ remains the authoritative
technical diagram.
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

CORE = """Draw a complete editorial scientific infographic, wide 16:9, pure white background, explaining an auditable thermal world model for a power-plant machine-learning paper.

STORY FLOW, left to right:
1. OBSERVED HISTORY card (far left): 96 time steps of plant sensor history drawn as thin horizontal timeline rows.
2. PROBABILISTIC OBSERVER card: compresses history into a probabilistic posterior state, drawn as a soft cyan cloud of uncertainty.
3. CENTER: ONE SHARED PHYSICS-STATE TRANSITION card -- THIS IS KEY -- one large prominent rounded card with a light thermal-process motif inside (thin heat-exchanger tube silhouettes). Inside, a small dashed divider showing the SAME physics block is reused for two purposes: prediction and action rollout.
4. PROBABILISTIC TEMPERATURE ROLLOUT card (right): 18 future temperature steps as a fading fan of uncertainty bands, pale green.
5. TOP PATH: a small LOGGED OR PROPOSED ACTIONS card feeding a pale amber ACTION-SUPPORT GATE card, then a solid arrow from the gate into the central transition card.
6. BOTTOM PATH: a small DECLARED BOUNDARY SCENARIO card feeding a pale teal BOUNDARY MODEL card, then a solid arrow into the central transition card.
7. BELOW CENTER: an ACTION-BLIND RESIDUAL CLOSURE card (pale teal) with a tiny padlock badge; a solid arrow up into the central transition.
8. ONE PROHIBITED PATH ONLY: a dashed RED arrow from a small gray chip labeled TRUE FUTURE BOUNDARY pointing toward the boundary model, crossed by one red X, with a tiny caption "forecast mode: REJECT".

STORY ELEMENTS:
- Top title band: "An Auditable Thermal World Model"
- Subtitle: "One shared physics-state transition drives prediction and action rollout -- with explicit information permissions"
- A thin 3-step strip below the main flow: "1. Encode 96-step history into a probabilistic state" / "2. Roll the same physics forward under a declared scenario" / "3. Simulate any logged or proposed action through the support gate"
- A small legend row: navy line = information flow, teal line = scenario / action, red dashed = prohibited
- Footer tagline: "Every path has a permission."

TEXT RULES: render ONLY the labels listed above and the module names below; no other words anywhere. Module name labels: "Observed history", "Probabilistic observer", "One shared physics-state transition", "Probabilistic temperature rollout", "Boundary model", "Action-support gate", "Action-blind residual closure".

STYLE: Nature-style editorial vector infographic; restrained navy outlines, soft cyan state modules, pale amber gate, pale teal closure and boundary, pale green output, red ONLY for the single prohibited path. Flat 2D rounded cards, thin precise non-overlapping connectors, generous whitespace, no gradients, no decorative icons, no 3D geometry, no axes, no dense microcopy, no invented mechanisms, no accuracy claims, no extra arrows."""

V1 = CORE

V2 = CORE.replace(
    "- Footer tagline: \"Every path has a permission.\"",
    """- Footer tagline: "Every path has a permission."
- Two small annotation callouts with thin leader lines: near the central transition "same physics, reused -- no separate counterfactual head"; near the closure "no action information enters this block".""")

for tag, prompt in (("v1_clean", V1), ("v2_narrative", V2)):
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
            continue
        images = d["choices"][0]["message"].get("images", [])
        for i, img in enumerate(images):
            url = img["image_url"]["url"]
            if url.startswith("data:"):
                b64 = url.split(",", 1)[1]
                path = OUT / (f"{tag}.png" if i == 0 else f"{tag}_{i}.png")
                path.write_bytes(base64.b64decode(b64))
                print(f"[{tag}] saved {path} ({path.stat().st_size} bytes) in {time.time()-t0:.0f}s")
        print(f"[{tag}] usage: {d.get('usage')}")
    except Exception as e:
        print(f"[{tag}] FAILED: {e}")
