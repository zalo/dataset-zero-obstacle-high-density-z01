# modal deploy scripts/deploy_api.py
#
# Serves the fine-tuned FLUX.2 Klein 4B model for PCB routing inference.
# Includes a test page at /test to visually compare model outputs against
# ground truth during training.
#
# Adapted from morphmaker.ai/morphmaker_api_flux2.py

import io
import random

import modal

app = modal.App("pcbrouter-flux2")

# Diffusers SHA: "Support Flux Klein peft (fal) lora format" (2026-02-21).
# Before the transformers v5 migration (2026-02-24) for stability.
GIT_SHA = "a80b19218b4bd4faf2d6d8c428dcf1ae6f11e43d"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.34.2",
        "datasets~=3.2.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]>=0.34.0",
        "peft>=0.17.0",
        "pillow>=10.0.0",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.51.0",
    )
    .apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/cache",
        }
    )
)

import base64
import json
import queue
import threading

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

with image.imports():
    import diffusers
    import torch
    from PIL import Image

FULL_MODEL_DIR = "/model-full"
FULL_OUTPUT_DIR = "/model-full/output"

# Klein base models are undistilled — use ~50 steps with CFG 4.0 for quality.
NUM_STEPS = 50

DEFAULT_INSTRUCTION = (
    "Route the traces between the color matched pins, using red for the top "
    "layer and blue for the bottom layer.  Add vias to keep traces of the same "
    "color from crossing."
)

# Test sample indices from the HF test split
TEST_SAMPLE_INDICES = [0, 500, 1000, 2000, 3000, 4000]
HF_DATASET = "tscircuit/zero-obstacle-high-density-z01"


def _sse_generate(pipe, prompt, input_image, strength, seed):
    """Run img2img pipeline in a thread, yield SSE events with progress."""
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    q = queue.Queue()

    # Effective steps for img2img = NUM_STEPS * strength
    effective_steps = max(1, int(NUM_STEPS * strength))

    def callback_on_step_end(pipe_self, step, timestep, callback_kwargs):
        progress = (step + 1) / effective_steps
        q.put(
            {
                "stage": "generating",
                "step": step + 1,
                "total": effective_steps,
                "progress": round(progress, 3),
            }
        )
        return callback_kwargs

    def run_pipeline():
        try:
            images = pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=NUM_STEPS,
                guidance_scale=4.0,
                height=256,
                width=256,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
            ).images
            with io.BytesIO() as buf:
                images[0].save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            q.put({"stage": "complete", "image": img_b64})
        except Exception as e:
            q.put({"stage": "error", "message": str(e)})
        finally:
            q.put(None)

    t = threading.Thread(target=run_pipeline)
    t.start()

    while True:
        event = q.get()
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"

    t.join()
    torch.cuda.empty_cache()


def _img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.cls(
    image=image,
    gpu="L4",
    timeout=5 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        FULL_MODEL_DIR: modal.Volume.from_name("pcbrouter-flux2-klein-full-volume"),
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
)
class Inference:
    @modal.enter()
    def initialize(self):
        import glob
        import os

        from datasets import load_dataset

        # Find the latest checkpoint by numeric sort (checkpoint-500, checkpoint-4500, etc.)
        checkpoints = glob.glob(os.path.join(FULL_OUTPUT_DIR, "checkpoint-*"))
        if checkpoints:
            checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = os.path.basename(checkpoint_path)
        else:
            checkpoint_path = None
            self.checkpoint_name = "final"

        # Load the base pipeline, then swap in the fine-tuned transformer.
        # Checkpoints only contain transformer weights, not a full pipeline.
        print(f"Loading base pipeline from: {FULL_MODEL_DIR}")
        self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            FULL_MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )

        if checkpoint_path:
            transformer_path = os.path.join(checkpoint_path, "transformer")
            if os.path.isdir(transformer_path):
                print(f"Loading fine-tuned transformer from: {transformer_path}")
                from diffusers import Flux2Transformer2DModel

                self.pipe.transformer = Flux2Transformer2DModel.from_pretrained(
                    transformer_path,
                    torch_dtype=torch.bfloat16,
                )
            else:
                print(f"Warning: no transformer dir in {checkpoint_path}, using base model")
        else:
            print(f"Loading final model from: {FULL_OUTPUT_DIR}")
            from diffusers import Flux2Transformer2DModel

            transformer_path = os.path.join(FULL_OUTPUT_DIR, "transformer")
            if os.path.isdir(transformer_path):
                self.pipe.transformer = Flux2Transformer2DModel.from_pretrained(
                    transformer_path,
                    torch_dtype=torch.bfloat16,
                )

        self.pipe.to("cuda")

        # Load test samples for the test page
        print("Loading test samples from HuggingFace...")
        ds = load_dataset(HF_DATASET, split="test")
        self.test_samples = []
        for idx in TEST_SAMPLE_INDICES:
            if idx < len(ds):
                row = ds[idx]
                self.test_samples.append(
                    {
                        "idx": idx,
                        "cond_b64": _img_to_b64(row["cond_image"]),
                        "gt_b64": _img_to_b64(row["output_image"]),
                    }
                )
        print(f"Loaded {len(self.test_samples)} test samples")

    @modal.fastapi_endpoint(method="GET", docs=True)
    async def status(self):
        """Return the current model checkpoint info."""
        return JSONResponse({"checkpoint": self.checkpoint_name})

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def route(self, request: Request):
        """Accept a connection-pairs image and return a routed image.

        POST body (JSON):
            input_image: base64-encoded PNG of the connection-pairs image
            instruction: (optional) edit instruction text
            strength:    (optional) img2img strength, 0.0-1.0 (default 0.75)
            seed:        (optional) RNG seed for reproducibility
        """
        body = await request.json()

        if "input_image" not in body:
            return JSONResponse(
                status_code=400,
                content={"error": "input_image is required"},
            )

        input_b64 = body["input_image"]
        instruction = body.get("instruction", DEFAULT_INSTRUCTION)
        strength = min(max(float(body.get("strength", 0.75)), 0.0), 1.0)
        seed = body.get("seed")

        input_bytes = base64.b64decode(input_b64)
        input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
        input_image = input_image.resize((256, 256))

        print(
            f"Inference: instruction={instruction!r}, "
            f"strength={strength}, seed={seed}, "
            f"image_size={input_image.size}"
        )

        return StreamingResponse(
            _sse_generate(self.pipe, instruction, input_image, strength, seed),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @modal.fastapi_endpoint(method="GET")
    async def test(self):
        """Test page to visually compare model outputs against ground truth."""
        samples_json = json.dumps(self.test_samples)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCB Router - Training Progress</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 8px; font-size: 1.4em; color: #fff; }}
  .subtitle {{ text-align: center; margin-bottom: 24px; color: #888; font-size: 0.9em; }}
  .controls {{ display: flex; gap: 12px; justify-content: center; align-items: center; margin-bottom: 24px; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.85em; color: #aaa; }}
  .controls input, .controls select {{ background: #1a1a1a; border: 1px solid #333; color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 0.85em; }}
  .controls input[type=range] {{ width: 120px; }}
  button {{ background: #2563eb; color: #fff; border: none; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9em; font-weight: 600; }}
  button:hover {{ background: #1d4ed8; }}
  button:disabled {{ background: #333; cursor: not-allowed; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr)); gap: 20px; }}
  .card {{ background: #141414; border: 1px solid #222; border-radius: 8px; padding: 16px; }}
  .card-header {{ font-size: 0.8em; color: #888; margin-bottom: 10px; }}
  .images {{ display: flex; gap: 8px; align-items: flex-start; }}
  .img-col {{ text-align: center; }}
  .img-col img {{ width: 200px; height: 200px; border-radius: 4px; background: #000; image-rendering: pixelated; }}
  .img-col .label {{ font-size: 0.7em; color: #666; margin-top: 4px; }}
  .progress {{ height: 3px; background: #222; border-radius: 2px; margin-top: 8px; overflow: hidden; }}
  .progress-fill {{ height: 100%; background: #2563eb; width: 0%; transition: width 0.3s; }}
  .status {{ font-size: 0.75em; color: #666; margin-top: 4px; text-align: right; }}
</style>
</head>
<body>
<h1>PCB Trace Router</h1>
<p class="subtitle">Checkpoint: <strong id="checkpoint">{self.checkpoint_name}</strong></p>

<div class="controls">
  <label>Strength: <input type="range" id="strength" min="0" max="1" step="0.05" value="0.75">
    <span id="strength-val">0.75</span></label>
  <label>Seed: <input type="number" id="seed" value="42" style="width:70px"></label>
  <button id="run-btn" onclick="runAll()">Generate All</button>
</div>

<div class="grid" id="grid"></div>

<script>
const SAMPLES = {samples_json};
const ROUTE_URL = window.location.origin.replace('-test', '-route');

document.getElementById('strength').addEventListener('input', e => {{
  document.getElementById('strength-val').textContent = e.target.value;
}});

function init() {{
  const grid = document.getElementById('grid');
  SAMPLES.forEach((s, i) => {{
    grid.innerHTML += `
      <div class="card" id="card-${{i}}">
        <div class="card-header">Test sample #${{s.idx}}</div>
        <div class="images">
          <div class="img-col">
            <img src="data:image/png;base64,${{s.cond_b64}}" />
            <div class="label">Input</div>
          </div>
          <div class="img-col">
            <img id="output-${{i}}" src="data:image/png;base64,${{s.gt_b64}}" style="opacity:0.3" />
            <div class="label" id="output-label-${{i}}">Ground Truth (faded)</div>
          </div>
          <div class="img-col">
            <img id="gt-${{i}}" src="data:image/png;base64,${{s.gt_b64}}" />
            <div class="label">Ground Truth</div>
          </div>
        </div>
        <div class="progress"><div class="progress-fill" id="prog-${{i}}"></div></div>
        <div class="status" id="status-${{i}}">Ready</div>
      </div>`;
  }});
}}

async function generate(idx) {{
  const s = SAMPLES[idx];
  const strength = parseFloat(document.getElementById('strength').value);
  const seed = parseInt(document.getElementById('seed').value);
  const outImg = document.getElementById('output-' + idx);
  const outLabel = document.getElementById('output-label-' + idx);
  const prog = document.getElementById('prog-' + idx);
  const status = document.getElementById('status-' + idx);

  outImg.style.opacity = '0.3';
  outLabel.textContent = 'Generating...';
  prog.style.width = '0%';
  status.textContent = 'Starting...';

  try {{
    const resp = await fetch(ROUTE_URL, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'Accept': 'text/event-stream' }},
      body: JSON.stringify({{ input_image: s.cond_b64, strength, seed }}),
    }});

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {{
      const {{ value, done }} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {{ stream: true }});
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {{
        if (!line.startsWith('data: ')) continue;
        try {{
          const event = JSON.parse(line.slice(6));
          if (event.stage === 'generating') {{
            const pct = Math.round(event.progress * 100);
            prog.style.width = pct + '%';
            status.textContent = `Step ${{event.step}}/${{event.total}}`;
          }} else if (event.stage === 'complete') {{
            outImg.src = 'data:image/png;base64,' + event.image;
            outImg.style.opacity = '1';
            outLabel.textContent = 'Model Output';
            prog.style.width = '100%';
            status.textContent = 'Done';
          }} else if (event.stage === 'error') {{
            status.textContent = 'Error: ' + event.message;
          }}
        }} catch (e) {{}}
      }}
    }}
  }} catch (e) {{
    status.textContent = 'Error: ' + e.message;
  }}
}}

async function runAll() {{
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  for (let i = 0; i < SAMPLES.length; i++) {{
    await generate(i);
  }}

  btn.disabled = false;
  btn.textContent = 'Generate All';
}}

init();
</script>
</body>
</html>"""
        return HTMLResponse(content=html)
